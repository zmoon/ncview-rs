use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use ndarray::Array2;
use oxinetcdf::{NcFile, NcGroup, NcType};

use super::slice::{CoordinateGrid, PackedAttributes, Slice2D, SliceRequest, classify_packed};
use super::{
    AxisRole, DataSource, DatasetFormat, DatasetMetadata, Dimension, PointCoordinates, Variable,
    normalize_longitude,
};
use crate::error::{NcvError, Result};

const HDF5_SIGNATURE: &[u8; 8] = b"\x89HDF\r\n\x1a\n";
const MAX_BYTES: usize = 768 * 1024 * 1024;

pub struct NetCdf4Source {
    path: PathBuf,
    file: NcFile,
    root: NcGroup,
    metadata: DatasetMetadata,
    coord_cache: Mutex<std::collections::HashMap<String, Vec<f64>>>,
}

impl NetCdf4Source {
    pub fn open(path: &Path) -> Result<Self> {
        let mut handle = File::open(path).map_err(|source| NcvError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let mut signature = [0_u8; 8];
        handle
            .read_exact(&mut signature)
            .map_err(|source| NcvError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        if &signature[0..3] == b"CDF" {
            return Err(NcvError::UnsupportedFormat {
                path: path.to_path_buf(),
                reason: "NetCDF-3 is unsupported in v0.1; convert to NetCDF-4".into(),
            });
        }
        if &signature != HDF5_SIGNATURE {
            return Err(NcvError::UnsupportedFormat {
                path: path.to_path_buf(),
                reason: "expected a NetCDF-4/HDF5 file".into(),
            });
        }
        let file = NcFile::open(path).map_err(|error| NcvError::InvalidDataset {
            path: path.to_path_buf(),
            reason: error.to_string(),
        })?;
        Self::from_file(path, file)
    }

    /// Open a bounded complete object supplied by a remote adapter.
    ///
    /// This is intentionally only used for explicitly bounded fallback objects. Large remote
    /// NetCDF-4 objects still require the source-backed OxiH5 reader seam.
    pub(crate) fn open_bytes(path: &Path, bytes: &[u8]) -> Result<Self> {
        validate_signature(path, bytes)?;
        let file = NcFile::open_from_bytes(bytes).map_err(|error| NcvError::InvalidDataset {
            path: path.to_path_buf(),
            reason: error.to_string(),
        })?;
        Self::from_file(path, file)
    }

    /// Open a source-backed NetCDF-4 object. `metadata` must contain the
    /// bounded HDF5 metadata window; payload and chunk-index reads are served
    /// through OxiH5's random-access source.
    pub(crate) fn open_source(
        path: &Path,
        metadata: Vec<u8>,
        source: Arc<dyn oxih5::ByteSource>,
    ) -> Result<Self> {
        validate_signature(path, &metadata)?;
        let file = NcFile::open_with_source(metadata, source).map_err(|error| {
            NcvError::InvalidDataset {
                path: path.to_path_buf(),
                reason: error.to_string(),
            }
        })?;
        Self::from_file(path, file)
    }

    fn from_file(path: &Path, file: NcFile) -> Result<Self> {
        let root = file
            .root_group()
            .map_err(|error| NcvError::InvalidDataset {
                path: path.to_path_buf(),
                reason: format!("missing NetCDF-4 conventions: {error}"),
            })?;
        let group_variables = collect_group_variables(&root);
        let mut dimensions: Vec<Dimension> = root
            .dimensions
            .iter()
            .map(|dimension| Dimension {
                name: dimension.name.clone(),
                length: usize::try_from(dimension.len).unwrap_or(usize::MAX),
                role: role_for_name(&dimension.name),
            })
            .collect();
        // Some libnetcdf files encode DIMENSION_LIST as a vlen array. The
        // low-level resolver intentionally falls back to phony dimensions for
        // that attribute, even though the corresponding 1-D coordinate
        // variables provide unambiguous canonical names.
        for entry in group_variables
            .iter()
            .filter(|entry| entry.variable.shape.len() == 1)
        {
            let variable = entry.variable;
            let role = axis_role(variable);
            if role == AxisRole::Other {
                continue;
            }
            let length = usize::try_from(variable.shape[0]).unwrap_or(usize::MAX);
            if let Some(raw_name) = variable.dim_names().first()
                && let Some(dimension) = dimensions.iter_mut().find(|d| d.name == *raw_name)
                && dimension.role == AxisRole::Other
            {
                dimension.role = role;
            }
            if !dimensions
                .iter()
                .any(|dimension| dimension.name == variable.name)
            {
                dimensions.push(Dimension {
                    name: variable.name.clone(),
                    length,
                    role,
                });
            }
        }
        let variables: Vec<Variable> = group_variables
            .iter()
            .map(|entry| Variable {
                name: entry.qualified_name.clone(),
                dimensions: canonical_dimension_names(
                    entry.variable,
                    &dimensions,
                    &coordinate_dimension_aliases(entry.group),
                ),
                units: entry.variable.units(),
                long_name: attribute_text(entry.variable, "long_name"),
                standard_name: attribute_text(entry.variable, "standard_name"),
                // Some libnetcdf/COARDS files use a datatype message variant that
                // oxih5 currently reports as `Opaque(<size>)` even though the
                // payload is a regular IEEE float. Keep multidimensional values
                // visible so the typed payload fallback below can decode them.
                numeric: (entry.variable.nc_type().is_numeric() || entry.variable.shape.len() >= 2)
                    && !is_coordinate_field(entry.variable),
            })
            .collect();
        let used_phony = variables
            .iter()
            .flat_map(|variable| variable.dimensions.iter())
            .filter(|name| name.starts_with("phony_dim_"))
            .cloned()
            .collect::<std::collections::HashSet<_>>();
        dimensions.retain(|dimension| {
            !dimension.name.starts_with("phony_dim_") || used_phony.contains(&dimension.name)
        });
        Ok(Self {
            path: path.to_path_buf(),
            file,
            root,
            metadata: DatasetMetadata {
                path: path.display().to_string(),
                format: DatasetFormat::NetCdf4,
                dimensions,
                variables,
            },
            coord_cache: Mutex::new(std::collections::HashMap::new()),
        })
    }
}

fn validate_signature(path: &Path, bytes: &[u8]) -> Result<()> {
    if bytes.len() < HDF5_SIGNATURE.len() {
        return Err(NcvError::UnsupportedFormat {
            path: path.to_path_buf(),
            reason: "object is too small to contain a NetCDF-4/HDF5 signature".into(),
        });
    }
    if &bytes[0..3] == b"CDF" {
        return Err(NcvError::UnsupportedFormat {
            path: path.to_path_buf(),
            reason: "NetCDF-3 is unsupported in v0.1; convert to NetCDF-4".into(),
        });
    }
    if &bytes[..HDF5_SIGNATURE.len()] != HDF5_SIGNATURE {
        return Err(NcvError::UnsupportedFormat {
            path: path.to_path_buf(),
            reason: "expected a NetCDF-4/HDF5 file".into(),
        });
    }
    Ok(())
}

impl DataSource for NetCdf4Source {
    fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }

    fn read_slice(&self, request: &SliceRequest) -> Result<Slice2D> {
        self.read_slice_on_axes(request, None, None, &[])
    }

    fn read_slice_on_axes(
        &self,
        request: &SliceRequest,
        row_axis_name: Option<&str>,
        col_axis_name: Option<&str>,
        fixed_axes: &[(String, usize)],
    ) -> Result<Slice2D> {
        let (group, variable) =
            self.find_variable(&request.variable)
                .ok_or_else(|| NcvError::UnsupportedVariable {
                    variable: request.variable.clone(),
                    reason: "variable not found".into(),
                })?;
        let shape = variable
            .shape
            .iter()
            .map(|size| {
                usize::try_from(*size)
                    .map_err(|_| NcvError::InvalidSlice("dimension exceeds usize".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        if shape.len() < 2 {
            return Err(NcvError::UnsupportedVariable {
                variable: request.variable.clone(),
                reason: "a plottable variable needs at least two dimensions".into(),
            });
        }
        let aliases = coordinate_dimension_aliases(group);
        let canonical_names =
            canonical_dimension_names(variable, &self.metadata.dimensions, &aliases);
        let canonical_refs = canonical_names
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let (row_axis, col_axis) = if let (Some(row), Some(col)) = (row_axis_name, col_axis_name)
            && let (Some(row_axis), Some(col_axis)) = (
                axis_index(&canonical_refs, row),
                axis_index(&canonical_refs, col),
            ) {
            (row_axis, col_axis)
        } else {
            spatial_axes(
                variable,
                group,
                &canonical_refs,
                shape.len() - 2,
                shape.len() - 1,
            )
        };
        if row_axis == col_axis {
            return Err(NcvError::UnsupportedVariable {
                variable: request.variable.clone(),
                reason: "display axes must be distinct".into(),
            });
        }
        let bounds = request.bounds;
        if bounds.row_end > shape[row_axis] || bounds.col_end > shape[col_axis] {
            return Err(NcvError::InvalidSlice(
                "slice bounds exceed variable shape".into(),
            ));
        }
        let dim_names = canonical_refs;
        let leading_axes = dim_names
            .iter()
            .enumerate()
            .filter(|(axis, _)| *axis != row_axis && *axis != col_axis)
            .map(|(axis, name)| (axis, *name))
            .collect::<Vec<_>>();
        let ranges = dim_names
            .iter()
            .enumerate()
            .map(|(axis, name)| {
                if axis == row_axis {
                    Ok(bounds.row_start..bounds.row_end)
                } else if axis == col_axis {
                    Ok(bounds.col_start..bounds.col_end)
                } else {
                    let inferred_role = leading_axes
                        .iter()
                        .position(|(leading_axis, _)| *leading_axis == axis)
                        .and_then(|position| match position {
                            0 => Some(AxisRole::Time),
                            1 => Some(AxisRole::Depth),
                            _ => None,
                        });
                    let role = match role_for_name(name) {
                        AxisRole::Other => inferred_role.unwrap_or(AxisRole::Other),
                        role => role,
                    };
                    let fixed_index = fixed_axes.iter().find_map(|(fixed_name, index)| {
                        fixed_name.eq_ignore_ascii_case(name).then_some(*index)
                    });
                    let index = match fixed_index {
                        Some(index) => index,
                        None => match role {
                            AxisRole::Time => request.time,
                            AxisRole::Depth => request.depth,
                            AxisRole::Other if shape[axis] == 1 => 0,
                            AxisRole::Other => {
                                return Err(NcvError::UnsupportedVariable {
                                    variable: request.variable.clone(),
                                    reason: format!(
                                        "non-spatial dimension '{name}' needs a time/depth role"
                                    ),
                                });
                            }
                            _ => 0,
                        },
                    };
                    if index >= shape[axis] {
                        return Err(NcvError::InvalidSlice(format!(
                            "index {index} exceeds dimension '{name}' length {}",
                            shape[axis]
                        )));
                    }
                    Ok(index..index + 1)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let output_elements = bounds.element_count()?;
        if output_elements
            .checked_mul(std::mem::size_of::<f64>())
            .is_none_or(|bytes| bytes > MAX_BYTES)
        {
            return Err(NcvError::InvalidSlice(
                "requested variable exceeds the 768 MiB working-set budget".into(),
            ));
        }
        let dataset = self
            .file
            .h5()
            .dataset_slice(&variable.h5_path, &ranges)
            .map_err(|error| NcvError::Adapter {
                path: self.path.clone(),
                reason: error.to_string(),
            })?;
        let raw =
            dataset_as_f64(&dataset, &variable.nc_type()).map_err(|error| NcvError::Adapter {
                path: self.path.clone(),
                reason: error.to_string(),
            })?;
        let expected = output_elements;
        if raw.len() != expected {
            return Err(NcvError::Adapter {
                path: self.path.clone(),
                reason: format!("slice returned {} values, expected {expected}", raw.len()),
            });
        }
        let (values, validity) = classify_packed(&raw, packed_attributes(variable));
        let values =
            Array2::from_shape_vec(bounds.shape(), values).map_err(|error| NcvError::Adapter {
                path: self.path.clone(),
                reason: error.to_string(),
            })?;
        let validity = Array2::from_shape_vec(bounds.shape(), validity).map_err(|error| {
            NcvError::Adapter {
                path: self.path.clone(),
                reason: error.to_string(),
            }
        })?;
        let slice = Slice2D::new(values, validity, bounds)?;
        let coordinates = if group.path == "/" {
            self.read_coordinate_grid(variable, &dim_names, &shape, row_axis, col_axis, bounds)
        } else {
            // Group-local coordinate references are not always exposed through
            // the root CF lookup table. The data slice remains fully usable;
            // source indices are retained until group-local coordinate support
            // is available.
            CoordinateGrid {
                latitude: None,
                longitude: None,
                latitude_axis: None,
                longitude_axis: None,
            }
        };
        Ok(slice.with_coordinates(coordinates))
    }

    fn time_label(&self, index: usize) -> Option<String> {
        NetCdf4Source::time_label(self, index)
    }

    fn vertical_label(&self, variable: &str, index: usize) -> Option<String> {
        NetCdf4Source::vertical_label(self, variable, index)
    }

    fn dimension_values(&self, variable: &str, dimension: &str) -> Option<Vec<f64>> {
        NetCdf4Source::dimension_values(self, variable, dimension)
    }

    fn point_coordinates(&self, variable: &str, row: usize, col: usize) -> PointCoordinates {
        NetCdf4Source::point_coordinates(self, variable, row, col)
    }
}

impl NetCdf4Source {
    pub(crate) fn read_variable_values(&self, requested: &str) -> Result<Vec<f64>> {
        let (_, variable) =
            self.find_variable(requested)
                .ok_or_else(|| NcvError::UnsupportedVariable {
                    variable: requested.to_string(),
                    reason: "variable not found".into(),
                })?;
        let shape = variable
            .shape
            .iter()
            .map(|size| {
                usize::try_from(*size)
                    .map_err(|_| NcvError::InvalidSlice("dimension exceeds usize".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let ranges = variable
            .shape
            .iter()
            .map(|&size| {
                let length = usize::try_from(size)
                    .map_err(|_| NcvError::InvalidSlice("dimension exceeds usize".into()))?;
                Ok(0..length)
            })
            .collect::<Result<Vec<_>>>()?;
        let dataset = self
            .file
            .h5()
            .dataset_slice(&variable.h5_path, &ranges)
            .map_err(|error| NcvError::Adapter {
                path: self.path.clone(),
                reason: error.to_string(),
            })?;
        let raw =
            dataset_as_f64(&dataset, &variable.nc_type()).map_err(|error| NcvError::Adapter {
                path: self.path.clone(),
                reason: error.to_string(),
            })?;
        let (values, _) = classify_packed(&raw, packed_attributes(variable));
        if values.len() != shape.iter().product::<usize>() {
            return Err(NcvError::Adapter {
                path: self.path.clone(),
                reason: format!(
                    "variable {} returned {} values, expected {}",
                    requested,
                    values.len(),
                    shape.iter().product::<usize>()
                ),
            });
        }
        Ok(values)
    }

    fn read_coordinate_values_cached(&self, variable: &oxinetcdf::NcVariable) -> Result<Vec<f64>> {
        if let Ok(cache) = self.coord_cache.lock()
            && let Some(cached) = cache.get(&variable.h5_path)
        {
            return Ok(cached.clone());
        }

        let element_count = variable
            .shape
            .iter()
            .try_fold(1usize, |acc, &len| {
                usize::try_from(len).ok()?.checked_mul(acc)
            })
            .ok_or_else(|| NcvError::InvalidSlice("coordinate shape exceeds usize".into()))?;

        let ranges = variable
            .shape
            .iter()
            .map(|&len| {
                let size = usize::try_from(len).map_err(|_| {
                    NcvError::InvalidSlice("coordinate dimension exceeds usize".into())
                })?;
                Ok(0..size)
            })
            .collect::<Result<Vec<_>>>()?;

        let raw = self
            .file
            .h5()
            .dataset_slice(&variable.h5_path, &ranges)
            .map_err(|error| NcvError::Adapter {
                path: self.path.clone(),
                reason: error.to_string(),
            })?;

        let values = decode_coordinate_values(&raw, variable)?;
        if values.len() == element_count
            && let Ok(mut cache) = self.coord_cache.lock()
        {
            cache.insert(variable.h5_path.clone(), values.clone());
        }
        Ok(values)
    }

    fn read_coordinate_grid(
        &self,
        variable: &oxinetcdf::NcVariable,
        data_names: &[&str],
        data_shape: &[usize],
        row_axis: usize,
        col_axis: usize,
        bounds: super::slice::Bounds,
    ) -> CoordinateGrid {
        let grid_shape = (data_shape[row_axis], data_shape[col_axis]);
        let declared = self.root.coordinates_of(&variable.name).unwrap_or_default();
        // Preserve the historical 2-D representation for small slices while
        // avoiding hundreds of MiB of duplicated coordinate data for large
        // regular rasters.
        let compact_axes = grid_shape
            .0
            .checked_mul(grid_shape.1)
            .is_some_and(|elements| elements > 1_000_000);
        let latitude_axis = compact_axes
            .then(|| {
                self.read_coordinate_axis(
                    &declared,
                    data_names,
                    data_shape,
                    AxisRole::Latitude,
                    row_axis,
                    grid_shape,
                    bounds,
                )
                .ok()
            })
            .flatten();
        let longitude_axis = compact_axes
            .then(|| {
                self.read_coordinate_axis(
                    &declared,
                    data_names,
                    data_shape,
                    AxisRole::Longitude,
                    col_axis,
                    grid_shape,
                    bounds,
                )
                .ok()
            })
            .flatten();
        CoordinateGrid {
            latitude: if latitude_axis.is_none() {
                self.read_coordinate_plane(
                    &declared,
                    data_names,
                    data_shape,
                    AxisRole::Latitude,
                    row_axis,
                    grid_shape,
                    bounds,
                )
                .ok()
            } else {
                None
            },
            longitude: if longitude_axis.is_none() {
                self.read_coordinate_plane(
                    &declared,
                    data_names,
                    data_shape,
                    AxisRole::Longitude,
                    col_axis,
                    grid_shape,
                    bounds,
                )
                .ok()
            } else {
                None
            },
            latitude_axis,
            longitude_axis,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn read_coordinate_axis(
        &self,
        declared: &[String],
        data_names: &[&str],
        data_shape: &[usize],
        role: AxisRole,
        axis: usize,
        grid_shape: (usize, usize),
        bounds: super::slice::Bounds,
    ) -> Result<Vec<f64>> {
        let coordinate =
            self.coordinate_variable(declared, data_names, data_shape, role, axis, grid_shape)?;
        if coordinate.shape.len() != 1 {
            return Err(NcvError::UnsupportedVariable {
                variable: coordinate.name.clone(),
                reason: "coordinate variable is not a 1-D axis".into(),
            });
        }
        let values = self.read_coordinate_values_cached(coordinate)?;
        let (start, end) = if role == AxisRole::Latitude {
            (bounds.row_start, bounds.row_end)
        } else {
            (bounds.col_start, bounds.col_end)
        };
        values.get(start..end).map(<[f64]>::to_vec).ok_or_else(|| {
            NcvError::InvalidSlice("coordinate bounds exceed coordinate length".into())
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn read_coordinate_plane(
        &self,
        declared: &[String],
        data_names: &[&str],
        data_shape: &[usize],
        role: AxisRole,
        axis: usize,
        grid_shape: (usize, usize),
        bounds: super::slice::Bounds,
    ) -> Result<Array2<f64>> {
        let coordinate =
            self.coordinate_variable(declared, data_names, data_shape, role, axis, grid_shape)?;
        let coordinate_shape = coordinate
            .shape
            .iter()
            .map(|size| {
                usize::try_from(*size).map_err(|_| {
                    NcvError::InvalidSlice("coordinate dimension exceeds usize".into())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let rows = bounds.row_end - bounds.row_start;
        let cols = bounds.col_end - bounds.col_start;
        let values = self.read_coordinate_values_cached(coordinate)?;
        if coordinate_shape.len() == 1 {
            let start = if role == AxisRole::Latitude {
                bounds.row_start
            } else {
                bounds.col_start
            };
            let end = if role == AxisRole::Latitude {
                bounds.row_end
            } else {
                bounds.col_end
            };
            if end > values.len() {
                return Err(NcvError::InvalidSlice(
                    "coordinate bounds exceed coordinate length".into(),
                ));
            }
            Ok(Array2::from_shape_fn((rows, cols), |(row, col)| {
                let idx = if role == AxisRole::Latitude {
                    start + row
                } else {
                    start + col
                };
                values[idx]
            }))
        } else if coordinate_shape.len() == 2 {
            let direct = coordinate_shape == [grid_shape.0, grid_shape.1];
            let transposed = coordinate_shape == [grid_shape.1, grid_shape.0];
            if !direct && !transposed {
                return Err(NcvError::InvalidSlice(
                    "2-D coordinate shape does not match data plane".into(),
                ));
            }
            let coord_cols = coordinate_shape[1];
            let plane = Array2::from_shape_fn((rows, cols), |(r, c)| {
                let (source_r, source_c) = if direct {
                    (bounds.row_start + r, bounds.col_start + c)
                } else {
                    (bounds.col_start + c, bounds.row_start + r)
                };
                let idx = source_r * coord_cols + source_c;
                values.get(idx).copied().unwrap_or(f64::NAN)
            });
            Ok(plane)
        } else {
            Err(NcvError::InvalidSlice(
                "coordinate variable is not 1-D or 2-D".into(),
            ))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn coordinate_variable<'a>(
        &'a self,
        declared: &[String],
        data_names: &[&str],
        data_shape: &[usize],
        role: AxisRole,
        axis: usize,
        grid_shape: (usize, usize),
    ) -> Result<&'a oxinetcdf::NcVariable> {
        let dimension_name = data_names.get(axis);
        declared
            .iter()
            .filter_map(|name| self.root.variable(name))
            .find(|candidate| axis_role(candidate) == role)
            .or_else(|| {
                self.root.variables.iter().find(|candidate| {
                    axis_role(candidate) == role
                        && candidate.shape.len() == 1
                        && candidate
                            .shape
                            .first()
                            .and_then(|size| usize::try_from(*size).ok())
                            == data_shape.get(axis).copied()
                })
            })
            .or_else(|| {
                dimension_name.and_then(|name| {
                    self.root
                        .variables
                        .iter()
                        .find(|candidate| candidate.name == *name && axis_role(candidate) == role)
                })
            })
            .or_else(|| {
                self.root.variables.iter().find(|candidate| {
                    axis_role(candidate) == role
                        && candidate.shape.len() == 2
                        && candidate
                            .shape
                            .iter()
                            .map(|size| usize::try_from(*size).ok())
                            .collect::<Option<Vec<_>>>()
                            .is_some_and(|shape| shape == [grid_shape.0, grid_shape.1])
                })
            })
            .ok_or_else(|| NcvError::UnsupportedVariable {
                variable: dimension_name
                    .map(|name| (*name).to_owned())
                    .unwrap_or_else(|| "coordinate".into()),
                reason: format!("no {role:?} coordinate variable found"),
            })
    }
}

impl NetCdf4Source {
    fn find_variable<'a>(
        &'a self,
        requested: &str,
    ) -> Option<(&'a NcGroup, &'a oxinetcdf::NcVariable)> {
        fn search<'a>(
            group: &'a NcGroup,
            requested: &str,
        ) -> Option<(&'a NcGroup, &'a oxinetcdf::NcVariable)> {
            let group_prefix = group.path.trim_matches('/');
            for variable in &group.variables {
                let qualified = if group_prefix.is_empty() {
                    variable.name.clone()
                } else {
                    format!("{group_prefix}/{}", variable.name)
                };
                if requested == variable.name || requested == qualified {
                    return Some((group, variable));
                }
            }
            group
                .children
                .iter()
                .find_map(|child| search(child, requested))
        }
        search(&self.root, requested)
    }

    fn time_label(&self, index: usize) -> Option<String> {
        let variable =
            self.root.variables.iter().find(|variable| {
                variable.shape.len() == 1 && axis_role(variable) == AxisRole::Time
            })?;
        let length = usize::try_from(*variable.shape.first()?).ok()?;
        if index >= length {
            return None;
        }
        let values = self.read_coordinate_values_cached(variable).ok()?;
        let value = *values.get(index)?;
        let units = variable.units();
        Some(format_time_coordinate(value, units.as_deref()))
    }

    fn vertical_label(&self, variable_name: &str, index: usize) -> Option<String> {
        let variable = self.root.variable(variable_name)?;
        let dimension_names = variable.dim_names();
        let declared = self.root.coordinates_of(variable_name).unwrap_or_default();
        let coordinate = declared
            .iter()
            .filter_map(|name| self.root.variable(name))
            .find(|candidate| axis_role(candidate) == AxisRole::Depth)
            .or_else(|| {
                dimension_names.iter().find_map(|dimension_name| {
                    self.root.variables.iter().find(|candidate| {
                        candidate.name == *dimension_name && axis_role(candidate) == AxisRole::Depth
                    })
                })
            })
            .or_else(|| {
                dimension_names.iter().find_map(|dimension_name| {
                    self.root.variables.iter().find(|candidate| {
                        axis_role(candidate) == AxisRole::Depth
                            && candidate.shape.len() == 1
                            && candidate.name.eq_ignore_ascii_case(dimension_name)
                    })
                })
            })?;
        let length = usize::try_from(*coordinate.shape.first()?).ok()?;
        if coordinate.shape.len() != 1 || index >= length {
            return None;
        }
        let values = self.read_coordinate_values_cached(coordinate).ok()?;
        let value = *values.get(index)?;
        if !value.is_finite() {
            return None;
        }
        let dimension_name = dimension_names
            .iter()
            .find(|name| coordinate.name.eq_ignore_ascii_case(name))
            .copied()
            .unwrap_or(coordinate.name.as_str());
        let description = attribute_text(coordinate, "long_name")
            .or_else(|| attribute_text(coordinate, "standard_name"))
            .unwrap_or_else(|| dimension_name.to_string());
        let units = coordinate.units().unwrap_or_default();
        let value_text = compact_coordinate_value(value);
        let level_text = if units.is_empty() {
            format!("{description} {value_text}")
        } else {
            format!("{description} {value_text} {units}")
        };
        Some(format!(
            "{level_text} (index {index} of {})",
            length.saturating_sub(1)
        ))
    }

    fn dimension_values(&self, variable_name: &str, dimension_name: &str) -> Option<Vec<f64>> {
        let variable = self.root.variable(variable_name)?;
        let declared = self.root.coordinates_of(variable_name).unwrap_or_default();
        let coordinate = declared
            .iter()
            .filter_map(|name| self.root.variable(name))
            .find(|candidate| {
                candidate.shape.len() == 1
                    && (candidate.name.eq_ignore_ascii_case(dimension_name)
                        || candidate
                            .dim_names()
                            .first()
                            .is_some_and(|name| name.eq_ignore_ascii_case(dimension_name)))
            })
            .or_else(|| {
                self.root.variables.iter().find(|candidate| {
                    candidate.shape.len() == 1
                        && candidate.name.eq_ignore_ascii_case(dimension_name)
                })
            })
            .or_else(|| {
                variable.dim_names().iter().find_map(|name| {
                    self.root.variables.iter().find(|candidate| {
                        candidate.shape.len() == 1
                            && candidate.name.eq_ignore_ascii_case(name)
                            && name.eq_ignore_ascii_case(dimension_name)
                    })
                })
            })?;
        self.read_coordinate_values_cached(coordinate).ok()
    }

    fn point_coordinates(&self, variable_name: &str, row: usize, col: usize) -> PointCoordinates {
        let Some(variable) = self.root.variable(variable_name) else {
            return PointCoordinates {
                latitude: None,
                longitude: None,
            };
        };
        let shape = variable
            .shape
            .iter()
            .map(|size| usize::try_from(*size).ok())
            .collect::<Option<Vec<_>>>();
        let Some(shape) = shape else {
            return PointCoordinates {
                latitude: None,
                longitude: None,
            };
        };
        let aliases = coordinate_dimension_aliases(&self.root);
        let names = canonical_dimension_names(variable, &self.metadata.dimensions, &aliases);
        let (row_axis, col_axis) = spatial_axes(
            variable,
            &self.root,
            &names.iter().map(String::as_str).collect::<Vec<_>>(),
            shape.len().saturating_sub(2),
            shape.len().saturating_sub(1),
        );
        let coordinates = self.root.coordinates_of(variable_name).unwrap_or_default();
        PointCoordinates {
            latitude: self.find_coordinate_value(
                &coordinates,
                &names,
                &shape,
                AxisRole::Latitude,
                row_axis,
                (shape[row_axis], shape[col_axis]),
                row,
                col,
            ),
            longitude: self
                .find_coordinate_value(
                    &coordinates,
                    &names,
                    &shape,
                    AxisRole::Longitude,
                    col_axis,
                    (shape[row_axis], shape[col_axis]),
                    row,
                    col,
                )
                .map(normalize_longitude),
        }
    }
}

impl NetCdf4Source {
    #[allow(clippy::too_many_arguments)]
    fn find_coordinate_value(
        &self,
        declared: &[String],
        data_names: &[String],
        data_shape: &[usize],
        role: AxisRole,
        axis: usize,
        grid_shape: (usize, usize),
        row: usize,
        col: usize,
    ) -> Option<f64> {
        let coordinate = declared
            .iter()
            .filter_map(|name| self.root.variable(name))
            .find(|coordinate| axis_role(coordinate) == role)
            .or_else(|| {
                self.root.variables.iter().find(|coordinate| {
                    axis_role(coordinate) == role
                        && coordinate.shape.len() == 1
                        && coordinate
                            .shape
                            .first()
                            .and_then(|size| usize::try_from(*size).ok())
                            == data_shape.get(axis).copied()
                })
            })
            .or_else(|| {
                let dimension_name = data_names.get(axis)?;
                self.root.variables.iter().find(|coordinate| {
                    coordinate.name == *dimension_name && axis_role(coordinate) == role
                })
            })?;
        let coordinate_shape = coordinate
            .shape
            .iter()
            .map(|size| usize::try_from(*size).ok())
            .collect::<Option<Vec<_>>>()?;
        let coordinate = if coordinate_shape.len() == 2 {
            coordinate
        } else {
            // Curvilinear files are not required to repeat the coordinate
            // names in a `coordinates` attribute. When metadata identifies a
            // 2-D latitude/longitude field, its grid shape is an unambiguous
            // fallback even if the HDF5 dimension names are opaque.
            self.root
                .variables
                .iter()
                .find(|candidate| {
                    axis_role(candidate) == role
                        && candidate.shape.len() == 2
                        && candidate
                            .shape
                            .iter()
                            .map(|size| usize::try_from(*size).ok())
                            .collect::<Option<Vec<_>>>()
                            .is_some_and(|shape| shape == [grid_shape.0, grid_shape.1])
                })
                .unwrap_or(coordinate)
        };
        let values = self.read_coordinate_values_cached(coordinate).ok()?;
        if coordinate.shape.len() == 1 {
            let index = if role == AxisRole::Latitude { row } else { col };
            values.get(index).copied().filter(|v| v.is_finite())
        } else if coordinate.shape.len() == 2 {
            let coordinate_rows = usize::try_from(coordinate.shape[0]).ok()?;
            let coordinate_cols = usize::try_from(coordinate.shape[1]).ok()?;
            let (coordinate_row, coordinate_col) =
                if (coordinate_rows, coordinate_cols) == grid_shape {
                    (row, col)
                } else if (coordinate_rows, coordinate_cols) == (grid_shape.1, grid_shape.0) {
                    (col, row)
                } else {
                    return None;
                };
            if coordinate_row >= coordinate_rows || coordinate_col >= coordinate_cols {
                return None;
            }
            let index = coordinate_row * coordinate_cols + coordinate_col;
            values.get(index).copied().filter(|v| v.is_finite())
        } else {
            None
        }
    }
}

#[derive(Clone, Copy)]
enum AxisKind {
    Latitude,
    Longitude,
}

fn spatial_axis(names: &[&str], kind: AxisKind, fallback: usize) -> usize {
    names
        .iter()
        .position(|name| match kind {
            AxisKind::Latitude => role_for_name(name) == AxisRole::Latitude,
            AxisKind::Longitude => role_for_name(name) == AxisRole::Longitude,
        })
        .unwrap_or(fallback)
}

fn axis_index(names: &[&str], requested: &str) -> Option<usize> {
    names.iter().position(|name| {
        name.eq_ignore_ascii_case(requested)
            || name
                .trim_start_matches("phony_dim_")
                .eq_ignore_ascii_case(requested.trim_start_matches("phony_dim_"))
    })
}

fn spatial_axes(
    variable: &oxinetcdf::NcVariable,
    root: &NcGroup,
    names: &[&str],
    row_fallback: usize,
    col_fallback: usize,
) -> (usize, usize) {
    let mut row = spatial_axis(names, AxisKind::Latitude, row_fallback);
    let mut col = spatial_axis(names, AxisKind::Longitude, col_fallback);
    if row != row_fallback || col != col_fallback {
        return (row, col);
    }

    // CF curvilinear coordinates carry their role on 2-D coordinate
    // variables, not on the dimensions themselves. Their dimension order is
    // the grid row/column order and is therefore the best source of truth.
    if let Some(coordinates) = root.coordinates_of(&variable.name) {
        for coordinate_name in coordinates {
            let Some(coordinate) = root.variable(&coordinate_name) else {
                continue;
            };
            if coordinate.shape.len() != 2 {
                continue;
            }
            let coordinate_dims = coordinate.dim_names();
            if coordinate_dims.len() != 2 {
                continue;
            }
            let Some(first) = names.iter().position(|name| *name == coordinate_dims[0]) else {
                continue;
            };
            let Some(second) = names.iter().position(|name| *name == coordinate_dims[1]) else {
                continue;
            };
            if first == second {
                continue;
            }
            match axis_role(coordinate) {
                AxisRole::Latitude => row = first,
                AxisRole::Longitude => col = second,
                _ => {}
            }
        }
    }
    (row, col)
}

fn canonical_dimension_names(
    variable: &oxinetcdf::NcVariable,
    dimensions: &[Dimension],
    aliases: &std::collections::HashMap<String, String>,
) -> Vec<String> {
    let original = variable.dim_names();
    let variable_role = axis_role(variable);
    if original.len() == 1 && variable_role != AxisRole::Other {
        return vec![variable.name.clone()];
    }
    // A 2-D latitude/longitude variable is a coordinate field, not a
    // dimension. Preserve its grid axes verbatim; otherwise it would create
    // misleading `lat`/`lon` dimensions in the sidebar.
    if original.len() >= 2 && variable_role != AxisRole::Other && has_coordinate_metadata(variable)
    {
        return original.iter().map(|name| (*name).to_owned()).collect();
    }
    let mut used = std::collections::HashSet::new();
    original
        .iter()
        .enumerate()
        .map(|(axis, name)| {
            if let Some(alias) = aliases.get(*name) {
                used.insert(alias.clone());
                return alias.clone();
            }
            if !name.starts_with("phony_dim_") {
                used.insert((*name).to_owned());
                return (*name).to_owned();
            }
            let role = if axis + 1 == original.len() {
                AxisRole::Longitude
            } else if axis + 2 == original.len() {
                AxisRole::Latitude
            } else if axis == 0 {
                AxisRole::Time
            } else if axis == 1 {
                AxisRole::Depth
            } else {
                AxisRole::Other
            };
            let length = variable.shape.get(axis).copied().unwrap_or_default();
            dimensions
                .iter()
                .filter(|dimension| {
                    dimension.role == role
                        && dimension.length == usize::try_from(length).unwrap_or(usize::MAX)
                        && !used.contains(&dimension.name)
                })
                .min_by_key(|dimension| dimension.name.starts_with("phony_dim_"))
                .map(|dimension| {
                    used.insert(dimension.name.clone());
                    dimension.name.clone()
                })
                .unwrap_or_else(|| (*name).to_owned())
        })
        .collect()
}

fn coordinate_dimension_aliases(root: &NcGroup) -> std::collections::HashMap<String, String> {
    let mut aliases = std::collections::HashMap::new();
    for variable in root
        .variables
        .iter()
        .filter(|variable| variable.shape.len() == 1)
    {
        if axis_role(variable) == AxisRole::Other {
            continue;
        }
        if let Some(raw_name) = variable.dim_names().first() {
            aliases
                .entry((*raw_name).to_owned())
                .or_insert_with(|| variable.name.clone());
        }
    }
    aliases
}

struct GroupVariable<'a> {
    group: &'a NcGroup,
    variable: &'a oxinetcdf::NcVariable,
    qualified_name: String,
}

fn collect_group_variables<'a>(root: &'a NcGroup) -> Vec<GroupVariable<'a>> {
    fn visit<'a>(group: &'a NcGroup, output: &mut Vec<GroupVariable<'a>>) {
        let prefix = group.path.trim_matches('/');
        for variable in &group.variables {
            let qualified_name = if prefix.is_empty() {
                variable.name.clone()
            } else {
                format!("{prefix}/{}", variable.name)
            };
            output.push(GroupVariable {
                group,
                variable,
                qualified_name,
            });
        }
        for child in &group.children {
            visit(child, output);
        }
    }
    let mut output = Vec::new();
    visit(root, &mut output);
    output
}

fn dataset_as_f64(dataset: &oxinetcdf::Dataset, kind: &NcType) -> Result<Vec<f64>> {
    let values = match kind {
        NcType::Float32 => dataset
            .as_f32()
            .map(|values| values.into_iter().map(f64::from).collect()),
        NcType::Float64 => dataset.as_f64(),
        NcType::Int8 => dataset
            .as_i8()
            .map(|values| values.into_iter().map(f64::from).collect()),
        NcType::Int16 => dataset
            .as_i16()
            .map(|values| values.into_iter().map(f64::from).collect()),
        NcType::Int32 => dataset
            .as_i32()
            .map(|values| values.into_iter().map(f64::from).collect()),
        NcType::Int64 => dataset
            .as_i64()
            .map(|values| values.into_iter().map(|value| value as f64).collect()),
        NcType::UInt8 => dataset
            .as_u8()
            .map(|values| values.into_iter().map(f64::from).collect()),
        NcType::UInt16 => dataset
            .as_u16()
            .map(|values| values.into_iter().map(f64::from).collect()),
        NcType::UInt32 => dataset
            .as_u32()
            .map(|values| values.into_iter().map(|value| value as f64).collect()),
        NcType::UInt64 => dataset
            .as_u64()
            .map(|values| values.into_iter().map(|value| value as f64).collect()),
        NcType::Opaque(_) => {
            let element_count = dataset.len();
            let element_size = dataset
                .data
                .len()
                .checked_div(element_count)
                .ok_or_else(|| NcvError::InvalidSlice("empty opaque dataset".into()))?;
            match element_size {
                4 => {
                    return Ok(dataset
                        .data
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|chunk| {
                            let bytes: [u8; 4] = *chunk;
                            f64::from(f32::from_le_bytes(bytes))
                        })
                        .collect::<Vec<_>>());
                }
                8 => {
                    return Ok(dataset
                        .data
                        .as_chunks::<8>()
                        .0
                        .iter()
                        .map(|chunk| {
                            let bytes: [u8; 8] = *chunk;
                            f64::from_le_bytes(bytes)
                        })
                        .collect::<Vec<_>>());
                }
                _ => {
                    return Err(NcvError::UnsupportedVariable {
                        variable: "requested variable".into(),
                        reason: format!(
                            "opaque element size {element_size} is not a float payload"
                        ),
                    });
                }
            }
        }
        _ => {
            return Err(NcvError::UnsupportedVariable {
                variable: "requested variable".into(),
                reason: format!("numeric type {kind:?} cannot be rendered"),
            });
        }
    };
    values.map_err(|error| NcvError::Adapter {
        path: PathBuf::new(),
        reason: error.to_string(),
    })
}

fn decode_coordinate_values(
    dataset: &oxinetcdf::Dataset,
    variable: &oxinetcdf::NcVariable,
) -> Result<Vec<f64>> {
    let raw = dataset_as_f64(dataset, &variable.nc_type())?;
    let (values, _) = classify_packed(&raw, packed_attributes(variable));
    Ok(values)
}

fn packed_attributes(variable: &oxinetcdf::NcVariable) -> PackedAttributes {
    let read = |name: &str| {
        variable.attr(name).and_then(|attribute| {
            attribute
                .as_f64()
                .ok()
                .and_then(|values| values.first().copied())
                .or_else(|| {
                    attribute
                        .as_i64()
                        .ok()
                        .and_then(|values| values.first().copied())
                        .map(|value| value as f64)
                })
                .or_else(|| attribute.raw().as_u64().map(|value| value as f64))
        })
    };
    PackedAttributes {
        fill: read("_FillValue"),
        missing: read("missing_value"),
        valid_min: read("valid_min"),
        valid_max: read("valid_max"),
        scale_factor: read("scale_factor"),
        add_offset: read("add_offset"),
    }
}

/// Resolve a CF/COARDS axis role from the variable metadata. Metadata is
/// deliberately preferred over names: files commonly call longitude `x`,
/// latitude `y`, or use generated/phony dimension names. Units are especially
/// useful for coordinate variables (`degrees_east`, `degrees_west`, etc.).
fn axis_role(variable: &oxinetcdf::NcVariable) -> AxisRole {
    if let Some(axis) = variable
        .attr("axis")
        .and_then(|attribute| attribute.as_text().ok())
        .and_then(|text| text.trim().chars().next())
    {
        match axis.to_ascii_uppercase() {
            'T' => return AxisRole::Time,
            'Z' => return AxisRole::Depth,
            'Y' => return AxisRole::Latitude,
            'X' => return AxisRole::Longitude,
            _ => {}
        }
    }
    if let Some(standard_name) = variable
        .attr("standard_name")
        .and_then(|attribute| attribute.as_text().ok())
    {
        match standard_name.trim().to_ascii_lowercase().as_str() {
            "time" | "forecast_reference_time" => return AxisRole::Time,
            "depth"
            | "altitude"
            | "height"
            | "ocean_sigma_coordinate"
            | "ocean_s_coordinate"
            | "ocean_s_coordinate_g1"
            | "ocean_s_coordinate_g2" => {
                return AxisRole::Depth;
            }
            "latitude" | "grid_latitude" => return AxisRole::Latitude,
            "longitude" | "grid_longitude" => return AxisRole::Longitude,
            _ => {}
        }
    }
    if let Some(units) = variable
        .attr("units")
        .and_then(|attribute| attribute.as_text().ok())
    {
        match normalized_units(&units).as_str() {
            "degreeeast" | "degreeseast" | "degreee" | "degreese" | "degreewest"
            | "degreeswest" | "degreew" | "degreesw" => return AxisRole::Longitude,
            "degreenorth" | "degreesnorth" | "degreen" | "degreesn" | "degreesouth"
            | "degreessouth" => return AxisRole::Latitude,
            value if value.contains("since") => return AxisRole::Time,
            _ => {}
        }
    }
    if variable.attr("positive").is_some() && variable.shape.len() == 1 {
        return AxisRole::Depth;
    }
    role_for_name(&variable.name)
}

fn normalized_units(units: &str) -> String {
    units
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase()
}

fn attribute_text(variable: &oxinetcdf::NcVariable, name: &str) -> Option<String> {
    variable
        .attr(name)
        .and_then(|attribute| attribute.as_text().ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn has_coordinate_metadata(variable: &oxinetcdf::NcVariable) -> bool {
    ["axis", "standard_name", "units"]
        .iter()
        .any(|name| variable.attr(name).is_some())
}

fn is_coordinate_field(variable: &oxinetcdf::NcVariable) -> bool {
    variable.shape.len() >= 2
        && has_coordinate_metadata(variable)
        && matches!(
            axis_role(variable),
            AxisRole::Latitude | AxisRole::Longitude
        )
}

fn role_for_name(name: &str) -> AxisRole {
    let lower = name.to_ascii_lowercase();
    if lower.contains("time") {
        AxisRole::Time
    } else if lower.contains("depth")
        || lower == "lev"
        || lower == "level"
        || lower == "levels"
        || lower.contains("pressure")
        || lower.contains("isobaric")
        || lower.contains("height")
        || lower.contains("altitude")
        || lower.contains("hybrid")
    {
        AxisRole::Depth
    } else if lower == "date" || lower == "dates" {
        AxisRole::Time
    } else if lower.contains("lat") {
        AxisRole::Latitude
    } else if lower.contains("lon") {
        AxisRole::Longitude
    } else {
        AxisRole::Other
    }
}

fn compact_coordinate_value(value: f64) -> String {
    let mut text = format!("{value:.6}");
    while text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('.') {
        text.pop();
    }
    text
}

fn format_time_coordinate(value: f64, units: Option<&str>) -> String {
    let Some(units) = units else {
        return format!("t={value}");
    };
    let Some((unit, origin)) = units.split_once(" since ") else {
        return format!("t={value} {units}");
    };
    let origin = origin.trim();
    let Some((year, month, day)) = parse_date(origin) else {
        return format!("t={value} {units}");
    };
    let multiplier = match unit.trim().to_ascii_lowercase().as_str() {
        "seconds" | "second" | "sec" | "s" => 1.0,
        "minutes" | "minute" | "min" => 60.0,
        "hours" | "hour" | "h" => 3_600.0,
        "days" | "day" | "d" => 86_400.0,
        _ => return format!("t={value} {units}"),
    };
    let total_seconds = (value * multiplier).round() as i64;
    let base_days = days_from_civil(year, month, day);
    let days = total_seconds.div_euclid(86_400);
    let seconds = total_seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(base_days + days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        seconds / 3_600,
        seconds % 3_600 / 60,
        seconds % 60
    )
}

fn parse_date(origin: &str) -> Option<(i64, i64, i64)> {
    let date = origin.get(..10)?;
    let mut parts = date.split('-');
    Some((
        parts.next()?.parse().ok()?,
        parts.next()?.parse().ok()?,
        parts.next()?.parse().ok()?,
    ))
}

fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = year - i64::from(month <= 2);
    let era = (if year >= 0 { year } else { year - 399 }) / 400;
    let year_of_era = year - era * 400;
    let month = month + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let days = days + 719_468;
    let era = (if days >= 0 { days } else { days - 146_096 }) / 146_097;
    let day_of_era = days - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    (year + i64::from(month <= 2), month, day)
}
