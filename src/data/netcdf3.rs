use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
};

use ndarray::Array2;
use netcdf3::{DataType, DataVector, FileReader};

use super::slice::{Slice2D, SliceRequest, Validity};
use super::{
    AxisRole, DataSource, DatasetFormat, DatasetMetadata, Dimension, PointCoordinates, Variable,
};
use crate::error::{NcvError, Result};

pub struct NetCdf3Source {
    path: PathBuf,
    metadata: DatasetMetadata,
    values: HashMap<String, Vec<f64>>,
}

pub fn is_netcdf3(path: &Path) -> bool {
    File::open(path)
        .and_then(|mut file| {
            let mut magic = [0_u8; 4];
            std::io::Read::read_exact(&mut file, &mut magic).map(|_| magic)
        })
        .is_ok_and(|magic| &magic[..3] == b"CDF")
}

impl NetCdf3Source {
    pub fn open(path: &Path) -> Result<Self> {
        let mut reader = FileReader::open(path).map_err(|error| NcvError::InvalidDataset {
            path: path.to_path_buf(),
            reason: error.to_string(),
        })?;
        let metadata = metadata(path, reader.data_set());
        let names = reader.data_set().get_var_names();
        let mut values = HashMap::with_capacity(names.len());
        for name in names {
            let data = reader.read_var(&name).map_err(|error| NcvError::Adapter {
                path: path.to_path_buf(),
                reason: error.to_string(),
            })?;
            values.insert(name, data_vector_as_f64(data));
        }
        Ok(Self {
            path: path.to_path_buf(),
            metadata,
            values,
        })
    }

    pub(crate) fn read_variable_values(&self, variable: &str) -> Result<Vec<f64>> {
        self.values
            .get(variable)
            .cloned()
            .ok_or_else(|| NcvError::UnsupportedVariable {
                variable: variable.to_string(),
                reason: "variable not found".into(),
            })
    }
}

impl DataSource for NetCdf3Source {
    fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }

    fn read_slice(&self, request: &SliceRequest) -> Result<Slice2D> {
        let variable = self
            .metadata
            .variables
            .iter()
            .find(|variable| variable.name == request.variable)
            .ok_or_else(|| NcvError::UnsupportedVariable {
                variable: request.variable.clone(),
                reason: "variable not found".into(),
            })?;
        if variable.dimensions.len() != 2 {
            return Err(NcvError::UnsupportedVariable {
                variable: request.variable.clone(),
                reason: "NetCDF-3 source currently supports 2-D variables; mesh variables use the MPAS flat-value path".into(),
            });
        }
        let rows = self
            .metadata
            .dimensions
            .iter()
            .find(|dimension| dimension.name == variable.dimensions[0])
            .map(|dimension| dimension.length)
            .unwrap_or(0);
        let cols = self
            .metadata
            .dimensions
            .iter()
            .find(|dimension| dimension.name == variable.dimensions[1])
            .map(|dimension| dimension.length)
            .unwrap_or(0);
        let bounds = request.bounds;
        if bounds.row_end > rows || bounds.col_end > cols {
            return Err(NcvError::InvalidSlice(
                "slice bounds exceed variable shape".into(),
            ));
        }
        let source_values = self.read_variable_values(&request.variable)?;
        if source_values.len() != rows.saturating_mul(cols) {
            return Err(NcvError::Adapter {
                path: self.path.clone(),
                reason: format!(
                    "variable returned {} values, expected {}",
                    source_values.len(),
                    rows * cols
                ),
            });
        }
        let values = Array2::from_shape_fn(bounds.shape(), |(row, col)| {
            source_values[(bounds.row_start + row) * cols + bounds.col_start + col]
        });
        let validity = Array2::from_shape_fn(bounds.shape(), |(_, _)| Validity::Finite);
        Slice2D::new(values, validity, bounds)
    }

    fn time_label(&self, _index: usize) -> Option<String> {
        None
    }

    fn vertical_label(&self, _variable: &str, _index: usize) -> Option<String> {
        None
    }

    fn dimension_values(&self, variable: &str, dimension: &str) -> Option<Vec<f64>> {
        let metadata = self
            .metadata
            .variables
            .iter()
            .find(|item| item.name == variable)?;
        if !metadata.dimensions.iter().any(|name| name == dimension) {
            return None;
        }
        self.read_variable_values(variable).ok()
    }

    fn point_coordinates(&self, _variable: &str, _row: usize, _col: usize) -> PointCoordinates {
        PointCoordinates::default()
    }
}

fn metadata(path: &Path, dataset: &netcdf3::DataSet) -> DatasetMetadata {
    let dimensions = dataset
        .get_dims()
        .into_iter()
        .map(|dimension| Dimension {
            name: dimension.name(),
            length: dimension.size(),
            role: role_for_name(&dimension.name()),
        })
        .collect::<Vec<_>>();
    let variables = dataset
        .get_vars()
        .into_iter()
        .map(|variable| {
            let name = variable.name().to_owned();
            Variable {
                name: name.clone(),
                dimensions: variable.dim_names(),
                numeric: !matches!(variable.data_type(), DataType::U8),
                units: dataset.get_var_attr_as_string(&name, "units"),
                long_name: dataset.get_var_attr_as_string(&name, "long_name"),
                standard_name: dataset.get_var_attr_as_string(&name, "standard_name"),
            }
        })
        .collect();
    DatasetMetadata {
        path: path.display().to_string(),
        format: DatasetFormat::NetCdf3,
        dimensions,
        variables,
    }
}

fn data_vector_as_f64(values: DataVector) -> Vec<f64> {
    match values {
        DataVector::I8(values) => values.into_iter().map(f64::from).collect(),
        DataVector::U8(values) => values.into_iter().map(f64::from).collect(),
        DataVector::I16(values) => values.into_iter().map(f64::from).collect(),
        DataVector::I32(values) => values.into_iter().map(f64::from).collect(),
        DataVector::F32(values) => values.into_iter().map(f64::from).collect(),
        DataVector::F64(values) => values,
    }
}

fn role_for_name(name: &str) -> AxisRole {
    let lower = name.to_ascii_lowercase();
    if lower.contains("time") || lower == "date" || lower == "dates" {
        AxisRole::Time
    } else if lower.contains("depth") || lower.contains("lev") || lower.contains("level") {
        AxisRole::Depth
    } else if lower.contains("lat") {
        AxisRole::Latitude
    } else if lower.contains("lon") {
        AxisRole::Longitude
    } else {
        AxisRole::Other
    }
}
