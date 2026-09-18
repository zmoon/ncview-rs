use std::path::{Path, PathBuf};

use crate::error::{NcvError, Result};

use super::{DataSource, DatasetMetadata, netcdf4::NetCdf4Source};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshLocation {
    Cell,
    Vertex,
}

impl MeshLocation {
    pub fn dimension_name(self) -> &'static str {
        match self {
            Self::Cell => "nCells",
            Self::Vertex => "nVertices",
        }
    }

    pub fn coordinate_names(self) -> (&'static str, &'static str) {
        match self {
            Self::Cell => ("latCell", "lonCell"),
            Self::Vertex => ("latVertex", "lonVertex"),
        }
    }
}

pub struct MpasSource {
    inner: Box<dyn DataSource>,
    mesh: MeshLocation,
    grid: Option<Box<dyn DataSource>>,
    path: PathBuf,
}

impl MpasSource {
    pub fn open(path: &Path, grid_path: Option<&Path>) -> Result<Box<dyn DataSource>> {
        let source = NetCdf4Source::open(path)?;
        let metadata = source.metadata().clone();
        let mesh = detect(&metadata).ok_or_else(|| {
            NcvError::InvalidDataset {
                path: path.to_path_buf(),
                reason: format!(
                    "MPAS mesh detection failed for {}; expected nCells or nVertices metadata",
                    path.display()
                ),
            }
        })?;

        let has_local_coordinates = local_coordinates_present(&metadata, mesh);
        if has_local_coordinates {
            return Ok(Box::new(Self {
                inner: Box::new(source),
                mesh,
                grid: None,
                path: path.to_path_buf(),
            }) as Box<dyn DataSource>);
        }

        let Some(grid_path) = grid_path.or_else(|| {
            metadata
                .variables
                .iter()
                .find_map(|variable| (variable.name == "latCell" || variable.name == "lonCell").then_some(path))
        }) else {
            return Err(NcvError::InvalidDataset {
                path: path.to_path_buf(),
                reason: format!(
                    "MPAS mesh detected ({} present) but lat/lon coordinates are not in this file; pass the init or static file with --grid <PATH>",
                    mesh.dimension_name()
                ),
            });
        };

        let grid_source = NetCdf4Source::open(grid_path)?;
        Ok(Box::new(Self {
            inner: Box::new(source),
            mesh,
            grid: Some(Box::new(grid_source) as Box<dyn DataSource>),
            path: path.to_path_buf(),
        }) as Box<dyn DataSource>)
    }
}

impl DataSource for MpasSource {
    fn metadata(&self) -> &DatasetMetadata {
        self.inner.metadata()
    }

    fn read_slice(&self, request: &super::slice::SliceRequest) -> Result<super::slice::Slice2D> {
        self.inner.read_slice(request)
    }

    fn read_slice_on_axes(
        &self,
        request: &super::slice::SliceRequest,
        row_axis: Option<&str>,
        col_axis: Option<&str>,
        fixed_axes: &[(String, usize)],
    ) -> Result<super::slice::Slice2D> {
        self.inner.read_slice_on_axes(request, row_axis, col_axis, fixed_axes)
    }

    fn time_label(&self, index: usize) -> Option<String> {
        self.inner.time_label(index)
    }

    fn vertical_label(&self, variable: &str, index: usize) -> Option<String> {
        self.inner.vertical_label(variable, index)
    }

    fn dimension_values(&self, variable: &str, dimension: &str) -> Option<Vec<f64>> {
        self.inner.dimension_values(variable, dimension)
    }

    fn point_coordinates(&self, _variable: &str, _row: usize, _col: usize) -> super::PointCoordinates {
        super::PointCoordinates::default()
    }
}

pub fn detect(metadata: &DatasetMetadata) -> Option<MeshLocation> {
    if metadata
        .dimensions
        .iter()
        .any(|dimension| dimension.name == MeshLocation::Cell.dimension_name())
    {
        return Some(MeshLocation::Cell);
    }
    if metadata
        .dimensions
        .iter()
        .any(|dimension| dimension.name == MeshLocation::Vertex.dimension_name())
    {
        return Some(MeshLocation::Vertex);
    }
    None
}

fn local_coordinates_present(metadata: &DatasetMetadata, mesh: MeshLocation) -> bool {
    let (lat_name, lon_name) = mesh.coordinate_names();
    metadata
        .variables
        .iter()
        .any(|variable| variable.name == lat_name || variable.name == lon_name)
}
