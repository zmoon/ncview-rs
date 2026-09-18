use std::path::{Path, PathBuf};

use ndarray::Array2;

use crate::analysis::projection::ProjectionIndex;
use crate::error::{NcvError, Result};

use super::{
    DataSource, DatasetMetadata,
    netcdf3::NetCdf3Source,
    netcdf4::NetCdf4Source,
    normalize_longitude,
    slice::{CoordinateGrid, Slice2D, Validity},
};

#[cfg(test)]
use super::slice::Bounds;

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
    inner: Box<dyn MeshValueSource>,
    grid: Option<Box<dyn MeshValueSource>>,
    path: PathBuf,
}

trait MeshValueSource: DataSource {
    fn read_variable_values(&self, variable: &str) -> Result<Vec<f64>>;

    fn read_mesh_values(
        &self,
        variable: &str,
        time: usize,
        depth: usize,
        mesh_dimension: &str,
    ) -> Result<Vec<f64>>;
}

impl MeshValueSource for NetCdf4Source {
    fn read_variable_values(&self, variable: &str) -> Result<Vec<f64>> {
        NetCdf4Source::read_variable_values(self, variable)
    }

    fn read_mesh_values(
        &self,
        variable: &str,
        time: usize,
        depth: usize,
        mesh_dimension: &str,
    ) -> Result<Vec<f64>> {
        let values = NetCdf4Source::read_variable_values(self, variable)?;
        select_mesh_values(
            &values,
            self.metadata(),
            variable,
            mesh_dimension,
            time,
            depth,
        )
    }
}

impl MeshValueSource for NetCdf3Source {
    fn read_variable_values(&self, variable: &str) -> Result<Vec<f64>> {
        NetCdf3Source::read_variable_values(self, variable)
    }

    fn read_mesh_values(
        &self,
        variable: &str,
        time: usize,
        depth: usize,
        mesh_dimension: &str,
    ) -> Result<Vec<f64>> {
        NetCdf3Source::read_mesh_values(self, variable, time, depth, mesh_dimension)
    }
}

fn radians_to_degrees(value: f64) -> f64 {
    value * 180.0 / std::f64::consts::PI
}

fn normalize_longitude_degrees(value: f64) -> f64 {
    normalize_longitude(value)
}

#[cfg(test)]
fn synthetic_window(
    lat: &[f64],
    lon: &[f64],
    values: &[f64],
    bounds: Bounds,
) -> (Vec<f64>, Vec<f64>, Array2<f64>) {
    let rows = bounds.row_end - bounds.row_start;
    let cols = bounds.col_end - bounds.col_start;
    let lat_min = lat
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f64::INFINITY, f64::min);
    let lat_max = lat
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    let lon_min = lon
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f64::INFINITY, f64::min);
    let lon_max = lon
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    let lat_axis: Vec<f64> = (0..rows)
        .map(|row| {
            let t = if rows == 1 {
                0.5
            } else {
                (row as f64 + 0.5) / rows as f64
            };
            lat_min + t * (lat_max - lat_min)
        })
        .collect();
    let lon_axis: Vec<f64> = (0..cols)
        .map(|col| {
            let t = if cols == 1 {
                0.5
            } else {
                (col as f64 + 0.5) / cols as f64
            };
            lon_min + t * (lon_max - lon_min)
        })
        .collect();
    let projection = ProjectionIndex::build(lat, lon, lat.len().max(1));
    let output = Array2::from_shape_fn((rows, cols), |(row, col)| {
        let latitude = lat_axis[row];
        let longitude = lon_axis[col];
        projection
            .nearest(latitude, longitude)
            .and_then(|source| {
                values
                    .get((source.row * lat.len().max(1)) + source.col)
                    .copied()
            })
            .unwrap_or(f64::NAN)
    });
    (lat_axis, lon_axis, output)
}

impl MpasSource {
    pub fn open(path: &Path, grid_path: Option<&Path>) -> Result<Box<dyn DataSource>> {
        let source = open_mesh_source(path)?;
        Self::from_source(path, source, grid_path)
    }

    fn from_source(
        path: &Path,
        source: Box<dyn MeshValueSource>,
        grid_path: Option<&Path>,
    ) -> Result<Box<dyn DataSource>> {
        let metadata = source.metadata().clone();
        let mesh = detect(&metadata).ok_or_else(|| NcvError::InvalidDataset {
            path: path.to_path_buf(),
            reason: format!(
                "MPAS mesh detection failed for {}; expected nCells or nVertices metadata",
                path.display()
            ),
        })?;

        let has_local_coordinates = local_coordinates_present(&metadata);
        if has_local_coordinates {
            return Ok(Box::new(Self {
                inner: source,
                grid: None,
                path: path.to_path_buf(),
            }) as Box<dyn DataSource>);
        }

        let Some(grid_path) = grid_path.or_else(|| {
            metadata.variables.iter().find_map(|variable| {
                (variable.name == "latCell" || variable.name == "lonCell").then_some(path)
            })
        }) else {
            return Err(NcvError::InvalidDataset {
                path: path.to_path_buf(),
                reason: format!(
                    "MPAS mesh detected ({} present) but lat/lon coordinates are not in this file; pass the init or static file with --grid <PATH>",
                    mesh.dimension_name()
                ),
            });
        };

        let grid_source = open_mesh_source(grid_path)?;
        Ok(Box::new(Self {
            inner: source,
            grid: Some(grid_source),
            path: path.to_path_buf(),
        }) as Box<dyn DataSource>)
    }

    fn coordinate_source(&self) -> &dyn MeshValueSource {
        self.grid.as_deref().unwrap_or(self.inner.as_ref())
    }

    fn read_mesh_coordinates(&self, mesh: MeshLocation) -> Result<(Vec<f64>, Vec<f64>)> {
        let (lat_name, lon_name) = mesh.coordinate_names();
        let source = self.coordinate_source();
        let lat = source.read_variable_values(lat_name)?;
        let lon = source.read_variable_values(lon_name)?;
        if lat.len() != lon.len() {
            return Err(NcvError::InvalidDataset {
                path: self.path.clone(),
                reason: format!(
                    "MPAS mesh coordinate lengths differ: {} lat values vs {} lon values",
                    lat.len(),
                    lon.len()
                ),
            });
        }
        let values = lat
            .iter()
            .zip(lon.iter())
            .filter_map(|(&lat_radians, &lon_radians)| {
                let latitude = radians_to_degrees(lat_radians);
                let longitude = normalize_longitude_degrees(radians_to_degrees(lon_radians));
                (latitude.is_finite() && longitude.is_finite()).then_some((latitude, longitude))
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();
        let (lat_deg, lon_deg) = values;
        Ok((lat_deg, lon_deg))
    }

    fn read_mesh_values_for_variable(
        &self,
        variable: &str,
        time: usize,
        depth: usize,
    ) -> Result<(MeshLocation, Vec<f64>)> {
        let mesh = self.mesh_for_variable(variable)?;
        self.inner
            .read_mesh_values(variable, time, depth, mesh.dimension_name())
            .map(|values| (mesh, values))
    }

    fn mesh_for_variable(&self, variable: &str) -> Result<MeshLocation> {
        let variable_metadata = self
            .inner
            .metadata()
            .variables
            .iter()
            .find(|item| item.name == variable)
            .ok_or_else(|| NcvError::UnsupportedVariable {
                variable: variable.to_owned(),
                reason: "variable not found".into(),
            })?;
        if variable_metadata
            .dimensions
            .iter()
            .any(|dimension| dimension == MeshLocation::Cell.dimension_name())
        {
            return Ok(MeshLocation::Cell);
        }
        if variable_metadata
            .dimensions
            .iter()
            .any(|dimension| dimension == MeshLocation::Vertex.dimension_name())
        {
            return Ok(MeshLocation::Vertex);
        }
        Err(NcvError::UnsupportedVariable {
            variable: variable.to_owned(),
            reason: "variable does not use nCells or nVertices".into(),
        })
    }
}

pub(crate) fn select_mesh_values(
    values: &[f64],
    metadata: &DatasetMetadata,
    variable: &str,
    mesh_dimension: &str,
    time: usize,
    depth: usize,
) -> Result<Vec<f64>> {
    let variable_metadata = metadata
        .variables
        .iter()
        .find(|item| item.name == variable)
        .ok_or_else(|| NcvError::UnsupportedVariable {
            variable: variable.to_owned(),
            reason: "variable not found".into(),
        })?;
    let dimensions = variable_metadata
        .dimensions
        .iter()
        .map(|name| {
            let length = metadata
                .dimensions
                .iter()
                .find(|dimension| dimension.name == *name)
                .map(|dimension| dimension.length)
                .ok_or_else(|| NcvError::InvalidDataset {
                    path: metadata.path.clone().into(),
                    reason: format!("variable '{variable}' references unknown dimension '{name}'"),
                })?;
            Ok((name.as_str(), length))
        })
        .collect::<Result<Vec<_>>>()?;
    let expected = dimensions
        .iter()
        .try_fold(1_usize, |product, (_, length)| product.checked_mul(*length))
        .ok_or_else(|| NcvError::InvalidSlice("MPAS variable shape overflows usize".into()))?;
    if expected != values.len() {
        return Err(NcvError::Adapter {
            path: metadata.path.clone().into(),
            reason: format!(
                "variable '{variable}' returned {} values, expected {expected}",
                values.len()
            ),
        });
    }
    let mesh_axis = dimensions
        .iter()
        .position(|(name, _)| *name == mesh_dimension)
        .ok_or_else(|| NcvError::UnsupportedVariable {
            variable: variable.to_owned(),
            reason: format!("variable does not use mesh dimension '{mesh_dimension}'"),
        })?;
    let mesh_length = dimensions[mesh_axis].1;
    let mut output = Vec::with_capacity(mesh_length);
    for mesh_index in 0..mesh_length {
        let mut linear_index = 0_usize;
        for (axis, (name, length)) in dimensions.iter().enumerate() {
            let index = if axis == mesh_axis {
                mesh_index
            } else if is_time_dimension(name) {
                time
            } else if is_vertical_dimension(name) {
                depth
            } else if *length == 1 {
                0
            } else {
                return Err(NcvError::UnsupportedVariable {
                    variable: variable.to_owned(),
                    reason: format!("non-spatial dimension '{name}' needs a time/depth role"),
                });
            };
            if index >= *length {
                return Err(NcvError::InvalidSlice(format!(
                    "index {index} exceeds dimension '{name}' length {length}"
                )));
            }
            let stride = dimensions
                .iter()
                .skip(axis + 1)
                .try_fold(1_usize, |product, (_, length)| product.checked_mul(*length))
                .ok_or_else(|| {
                    NcvError::InvalidSlice("MPAS variable stride overflows usize".into())
                })?;
            linear_index = linear_index
                .checked_add(index.checked_mul(stride).ok_or_else(|| {
                    NcvError::InvalidSlice("MPAS variable index overflows usize".into())
                })?)
                .ok_or_else(|| {
                    NcvError::InvalidSlice("MPAS variable index overflows usize".into())
                })?;
        }
        output.push(values[linear_index]);
    }
    Ok(output)
}

pub(crate) fn is_time_dimension(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("time") || lower == "date" || lower == "dates"
}

pub(crate) fn is_vertical_dimension(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("depth")
        || lower.contains("level")
        || lower.contains("lev")
        || lower.contains("pressure")
        || lower.contains("height")
        || lower.contains("altitude")
}

fn open_mesh_source(path: &Path) -> Result<Box<dyn MeshValueSource>> {
    if super::netcdf3::is_netcdf3(path) {
        Ok(Box::new(NetCdf3Source::open(path)?) as Box<dyn MeshValueSource>)
    } else {
        Ok(Box::new(NetCdf4Source::open(path)?) as Box<dyn MeshValueSource>)
    }
}

impl DataSource for MpasSource {
    fn metadata(&self) -> &DatasetMetadata {
        self.inner.metadata()
    }

    fn read_slice(&self, request: &super::slice::SliceRequest) -> Result<super::slice::Slice2D> {
        let (mesh, mesh_values) =
            self.read_mesh_values_for_variable(&request.variable, request.time, request.depth)?;
        let (lat_deg, lon_deg) = self.read_mesh_coordinates(mesh)?;
        if lat_deg.len() != mesh_values.len() {
            return Err(NcvError::InvalidDataset {
                path: self.path.clone(),
                reason: format!(
                    "MPAS mesh length mismatch for {}: {} coordinate values, {} data values",
                    request.variable,
                    lat_deg.len(),
                    mesh_values.len()
                ),
            });
        }

        let rows = request
            .bounds
            .row_end
            .saturating_sub(request.bounds.row_start);
        let cols = request
            .bounds
            .col_end
            .saturating_sub(request.bounds.col_start);
        if rows == 0 || cols == 0 {
            return Err(NcvError::InvalidSlice(
                "requested MPAS window is empty".into(),
            ));
        }

        let lat_min = lat_deg
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(f64::INFINITY, f64::min);
        let lat_max = lat_deg
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        let lon_min = lon_deg
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(f64::INFINITY, f64::min);
        let lon_max = lon_deg
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);

        let mut latitude_axis = Vec::with_capacity(rows);
        let mut longitude_axis = Vec::with_capacity(cols);
        if lat_min.is_finite() && lat_max.is_finite() && lon_min.is_finite() && lon_max.is_finite()
        {
            for row in 0..rows {
                let t = if rows == 1 {
                    0.5
                } else {
                    (row as f64 + 0.5) / rows as f64
                };
                latitude_axis.push(lat_min + t * (lat_max - lat_min));
            }
            for col in 0..cols {
                let t = if cols == 1 {
                    0.5
                } else {
                    (col as f64 + 0.5) / cols as f64
                };
                longitude_axis.push(lon_min + t * (lon_max - lon_min));
            }
        }

        let projection = ProjectionIndex::build(&lat_deg, &lon_deg, lat_deg.len().max(1));
        let values = Array2::from_shape_fn((rows, cols), |(row, col)| {
            let latitude = latitude_axis.get(row).copied().unwrap_or(f64::NAN);
            let longitude = longitude_axis.get(col).copied().unwrap_or(f64::NAN);
            projection
                .nearest(latitude, longitude)
                .and_then(|source| {
                    let source_index = source.row * lat_deg.len().max(1) + source.col;
                    mesh_values.get(source_index).copied()
                })
                .unwrap_or(f64::NAN)
        });
        let validity = Array2::from_shape_fn((rows, cols), |(_, _)| Validity::Finite);
        let mut slice = Slice2D::new(values.clone(), validity, request.bounds)?;
        slice = slice.with_coordinates(CoordinateGrid {
            latitude: None,
            longitude: None,
            latitude_axis: if latitude_axis.is_empty() {
                None
            } else {
                Some(latitude_axis)
            },
            longitude_axis: if longitude_axis.is_empty() {
                None
            } else {
                Some(longitude_axis)
            },
        });
        Ok(slice)
    }

    fn read_slice_on_axes(
        &self,
        request: &super::slice::SliceRequest,
        row_axis: Option<&str>,
        col_axis: Option<&str>,
        fixed_axes: &[(String, usize)],
    ) -> Result<super::slice::Slice2D> {
        let _ = (row_axis, col_axis, fixed_axes);
        self.read_slice(request)
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

    fn point_coordinates(
        &self,
        _variable: &str,
        _row: usize,
        _col: usize,
    ) -> super::PointCoordinates {
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

fn local_coordinates_present(metadata: &DatasetMetadata) -> bool {
    [MeshLocation::Cell, MeshLocation::Vertex]
        .into_iter()
        .any(|mesh| {
            let (lat_name, lon_name) = mesh.coordinate_names();
            metadata
                .variables
                .iter()
                .any(|variable| variable.name == lat_name)
                && metadata
                    .variables
                    .iter()
                    .any(|variable| variable.name == lon_name)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radians_to_degrees_and_wrap_longitude() {
        let latitude = radians_to_degrees(0.5);
        let longitude = normalize_longitude_degrees(540.0);

        assert!((latitude - 28.64788975654116).abs() < 1e-9);
        assert!((longitude - (-180.0)).abs() < 1e-9);
    }

    #[test]
    fn resamples_to_requested_window() {
        let mesh_lat = vec![0.0, 0.0, 0.0, 0.0];
        let mesh_lon = vec![0.0, 90.0, 180.0, -90.0];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let bounds = Bounds::new(0, 2, 0, 2).unwrap();
        let (lat_axis, lon_axis, output) = synthetic_window(&mesh_lat, &mesh_lon, &values, bounds);

        assert_eq!(output.nrows(), 2);
        assert_eq!(output.ncols(), 2);
        assert_eq!(lat_axis.len(), 2);
        assert_eq!(lon_axis.len(), 2);
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn selects_time_and_vertical_plane_for_mesh_dimension() {
        let metadata = DatasetMetadata {
            path: "fixture.nc".into(),
            format: super::super::DatasetFormat::NetCdf3,
            dimensions: vec![
                super::super::Dimension {
                    name: "Time".into(),
                    length: 2,
                    role: super::super::AxisRole::Time,
                },
                super::super::Dimension {
                    name: "nVertices".into(),
                    length: 3,
                    role: super::super::AxisRole::Other,
                },
                super::super::Dimension {
                    name: "nVertLevels".into(),
                    length: 2,
                    role: super::super::AxisRole::Depth,
                },
            ],
            variables: vec![super::super::Variable {
                name: "field".into(),
                dimensions: vec!["Time".into(), "nVertices".into(), "nVertLevels".into()],
                numeric: true,
                units: None,
                long_name: None,
                standard_name: None,
            }],
        };
        let values = (0..12).map(f64::from).collect::<Vec<_>>();

        let selected = select_mesh_values(&values, &metadata, "field", "nVertices", 1, 0).unwrap();

        assert_eq!(selected, [6.0, 8.0, 10.0]);
    }
}
