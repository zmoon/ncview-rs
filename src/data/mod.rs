//! Read-only dataset and slice abstractions.

pub mod coordinates;
pub mod fixtures;
pub mod grib2;
pub mod grib2_catalog;
pub mod grib2_identity;
pub mod grib2_index;
pub mod grib2_manifest;
pub mod grib2_types;
pub mod mpas;
pub mod netcdf3;
pub mod netcdf4;
pub mod remote;
pub mod remote_grib2;
pub mod remote_hdf5;
pub mod remote_netcdf4;
pub mod slice;
pub mod virtual_dataset;

use std::{
    fs::File,
    io::Read,
    path::Path,
    sync::{Arc, atomic::AtomicBool},
};

use crate::error::Result;
use crate::storage::location::SourceLocation;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetFormat {
    NetCdf3,
    NetCdf4,
    Grib2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisRole {
    Time,
    Depth,
    Latitude,
    Longitude,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dimension {
    pub name: String,
    pub length: usize,
    pub role: AxisRole,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Variable {
    pub name: String,
    pub dimensions: Vec<String>,
    pub numeric: bool,
    pub units: Option<String>,
    pub long_name: Option<String>,
    pub standard_name: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PointCoordinates {
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
}

/// Normalize geographic longitudes to the convention used in the viewer.
/// Non-finite values are preserved so missing coordinate diagnostics are not
/// turned into arbitrary geographic positions.
pub fn normalize_longitude(value: f64) -> f64 {
    if value.is_finite() {
        (value + 180.0).rem_euclid(360.0) - 180.0
    } else {
        value
    }
}

pub fn is_mesh_variable(variable: &Variable) -> bool {
    variable.dimensions.iter().any(|dimension| {
        dimension.eq_ignore_ascii_case("nCells") || dimension.eq_ignore_ascii_case("nVertices")
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetMetadata {
    pub path: String,
    pub format: DatasetFormat,
    pub dimensions: Vec<Dimension>,
    pub variables: Vec<Variable>,
}

pub trait DataSource: Send + Sync {
    fn metadata(&self) -> &DatasetMetadata;
    fn read_slice(&self, request: &slice::SliceRequest) -> Result<slice::Slice2D>;

    fn is_remote(&self) -> bool {
        false
    }

    /// Stable source-version token for remote collection diagnostics and cache identity.
    fn source_identity(&self) -> Option<&str> {
        None
    }

    fn remote_capabilities(&self) -> Option<remote::AccessCapabilities> {
        None
    }

    fn read_slice_on_axes(
        &self,
        request: &slice::SliceRequest,
        _row_axis: Option<&str>,
        _col_axis: Option<&str>,
        _fixed_axes: &[(String, usize)],
    ) -> Result<slice::Slice2D> {
        self.read_slice(request)
    }

    /// Variant used by interactive workers that may supersede an in-flight
    /// provider operation. Format adapters that can propagate cancellation
    /// override this; local sources retain the ordinary synchronous behavior.
    fn read_slice_on_axes_cancellable(
        &self,
        request: &slice::SliceRequest,
        row_axis: Option<&str>,
        col_axis: Option<&str>,
        fixed_axes: &[(String, usize)],
        _cancelled: Arc<AtomicBool>,
    ) -> Result<slice::Slice2D> {
        self.read_slice_on_axes(request, row_axis, col_axis, fixed_axes)
    }

    fn time_label(&self, _index: usize) -> Option<String> {
        None
    }

    fn time_label_for_variable(&self, _variable: &str, index: usize) -> Option<String> {
        self.time_label(index)
    }

    fn vertical_label(&self, _variable: &str, _index: usize) -> Option<String> {
        None
    }

    /// Labels for every index of the variable's vertical (Depth) axis, in
    /// index order. Empty when the variable has no vertical axis. The sidebar
    /// level list renders this, so it is read once per variable change rather
    /// than per frame.
    fn vertical_labels(&self, variable: &str) -> Vec<String> {
        let metadata = self.metadata();
        let count = metadata
            .variables
            .iter()
            .find(|item| item.name == variable)
            .and_then(|item| {
                item.dimensions.iter().find_map(|name| {
                    metadata
                        .dimensions
                        .iter()
                        .find(|dimension| &dimension.name == name)
                        .filter(|dimension| dimension.role == AxisRole::Depth)
                        .map(|dimension| dimension.length)
                })
            })
            .unwrap_or(0);
        (0..count)
            .filter_map(|index| self.vertical_label(variable, index))
            .collect()
    }

    fn dimension_values(&self, _variable: &str, _dimension: &str) -> Option<Vec<f64>> {
        None
    }

    fn point_coordinates(&self, _variable: &str, _row: usize, _col: usize) -> PointCoordinates {
        PointCoordinates {
            latitude: None,
            longitude: None,
        }
    }
}

pub fn open(path: impl AsRef<Path>) -> Result<Box<dyn DataSource>> {
    let path = path.as_ref();
    open_with_grid(path, None::<&Path>)
}

pub fn open_with_grid(
    path: impl AsRef<Path>,
    grid_path: Option<&Path>,
) -> Result<Box<dyn DataSource>> {
    let path = path.as_ref();
    let extension_matches = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "grib" | "grib2" | "grb" | "grb2"
            )
        });
    let magic_matches = File::open(path)
        .and_then(|mut file| {
            let mut magic = [0_u8; 4];
            file.read_exact(&mut magic).map(|_| magic)
        })
        .is_ok_and(|magic| magic == *b"GRIB");
    if extension_matches || magic_matches {
        grib2::Grib2Source::open(path).map(|source| Box::new(source) as Box<dyn DataSource>)
    } else if netcdf3::is_netcdf3(path) {
        let source = netcdf3::NetCdf3Source::open(path)?;
        if mpas::detect(source.metadata()).is_some() {
            return mpas::MpasSource::open(path, grid_path);
        }
        Ok(Box::new(source) as Box<dyn DataSource>)
    } else {
        let source = netcdf4::NetCdf4Source::open(path)?;
        if let Some(mesh) = mpas::detect(source.metadata()) {
            let mut path_for_grid = grid_path;
            if path_for_grid.is_none() {
                path_for_grid = Some(path);
            }
            let wrapped = mpas::MpasSource::open(path, path_for_grid)
                .map(|source| source as Box<dyn DataSource>)?;
            if matches!(mesh, mpas::MeshLocation::Cell | mpas::MeshLocation::Vertex) {
                return Ok(wrapped);
            }
        }
        Ok(Box::new(source) as Box<dyn DataSource>)
    }
}

/// Open either a local path or an explicit cloud object location.
pub fn open_location(location: impl AsRef<str>) -> Result<Box<dyn DataSource>> {
    open_location_with_progress(location, &|_| true)
}

pub fn open_location_with_grid(
    location: impl AsRef<str>,
    grid: Option<&str>,
) -> Result<Box<dyn DataSource>> {
    let source = SourceLocation::parse(location.as_ref())?;
    if source.is_remote() {
        remote::open_remote_with_progress(source, &|_| true)
    } else {
        let path = source.local_path().expect("local source has a path");
        let grid_path = grid.map(Path::new);
        open_with_grid(path, grid_path)
    }
}

pub fn open_location_with_progress(
    location: impl AsRef<str>,
    progress: &dyn Fn(&str) -> bool,
) -> Result<Box<dyn DataSource>> {
    let source = SourceLocation::parse(location.as_ref())?;
    if source.is_remote() {
        remote::open_remote_with_progress(source, progress)
    } else {
        open(source.local_path().expect("local source has a path"))
    }
}
