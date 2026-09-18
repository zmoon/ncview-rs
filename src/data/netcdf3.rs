use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::Mutex,
};

use ndarray::Array2;
use netcdf_reader::{NcAttrValue, NcFile, NcType};

use super::slice::{Slice2D, SliceRequest, Validity};
use super::{
    AxisRole, DataSource, DatasetFormat, DatasetMetadata, Dimension, PointCoordinates, Variable,
};
use crate::error::{NcvError, Result};

pub struct NetCdf3Source {
    path: PathBuf,
    metadata: DatasetMetadata,
    reader: Mutex<NcFile>,
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
        let reader = NcFile::open(path).map_err(|error| NcvError::InvalidDataset {
            path: path.to_path_buf(),
            reason: error.to_string(),
        })?;
        let metadata = metadata(path, &reader)?;
        Ok(Self {
            path: path.to_path_buf(),
            metadata,
            reader: Mutex::new(reader),
        })
    }

    pub(crate) fn read_variable_values(&self, variable: &str) -> Result<Vec<f64>> {
        let reader = self.reader.lock().map_err(|_| NcvError::Adapter {
            path: self.path.clone(),
            reason: "NetCDF-3 reader lock was poisoned".into(),
        })?;
        let values = reader
            .read_variable_as_f64(variable)
            .map_err(|error| NcvError::Adapter {
                path: self.path.clone(),
                reason: error.to_string(),
            })?;
        Ok(values.into_iter().collect())
    }

    pub(crate) fn read_mesh_values(
        &self,
        variable: &str,
        time: usize,
        depth: usize,
        mesh_dimension: &str,
    ) -> Result<Vec<f64>> {
        let reader = self.reader.lock().map_err(|_| NcvError::Adapter {
            path: self.path.clone(),
            reason: "NetCDF-3 reader lock was poisoned".into(),
        })?;
        let variables = reader.variables().map_err(|error| NcvError::Adapter {
            path: self.path.clone(),
            reason: error.to_string(),
        })?;
        let selected = variables
            .iter()
            .find(|candidate| candidate.name == variable)
            .ok_or_else(|| NcvError::UnsupportedVariable {
                variable: variable.to_owned(),
                reason: "variable not found".into(),
            })?;
        if !selected.is_record_var {
            let values =
                reader
                    .read_variable_as_f64(variable)
                    .map_err(|error| NcvError::Adapter {
                        path: self.path.clone(),
                        reason: error.to_string(),
                    })?;
            return super::mpas::select_mesh_values(
                &values.into_iter().collect::<Vec<_>>(),
                &self.metadata,
                variable,
                mesh_dimension,
                time,
                depth,
            );
        }

        let element_size = selected.dtype.size().map_err(|error| NcvError::Adapter {
            path: self.path.clone(),
            reason: error.to_string(),
        })?;
        let record_bytes = usize::try_from(selected.record_size).map_err(|_| {
            NcvError::InvalidSlice("NetCDF-3 record size exceeds platform usize".into())
        })?;
        let record_stride = record_stride(variables)?;
        let variable_offset = variables
            .iter()
            .filter(|candidate| candidate.is_record_var)
            .take_while(|candidate| candidate.name != selected.name)
            .try_fold(0_u64, |offset, candidate| {
                offset.checked_add(padded_record_size(candidate.record_size))
            })
            .ok_or_else(|| NcvError::InvalidSlice("NetCDF-3 record offset overflows".into()))?;
        let offset =
            selected
                .data_offset
                .checked_add((time as u64).checked_mul(record_stride).ok_or_else(|| {
                    NcvError::InvalidSlice("NetCDF-3 record offset overflows".into())
                })?)
                .and_then(|offset| offset.checked_add(variable_offset))
                .ok_or_else(|| NcvError::InvalidSlice("NetCDF-3 record offset overflows".into()))?;
        let mut file = File::open(&self.path).map_err(|source| NcvError::Io {
            path: self.path.clone(),
            source,
        })?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|source| NcvError::Io {
                path: self.path.clone(),
                source,
            })?;
        let mut raw = vec![0_u8; record_bytes];
        file.read_exact(&mut raw).map_err(|source| NcvError::Io {
            path: self.path.clone(),
            source,
        })?;
        let values = decode_numeric(&raw, &selected.dtype, record_bytes / element_size)?;
        super::mpas::select_mesh_values(
            &values,
            &self.metadata,
            variable,
            mesh_dimension,
            time,
            depth,
        )
    }
}

fn padded_record_size(size: u64) -> u64 {
    let remainder = size % 4;
    if remainder == 0 {
        size
    } else {
        size + 4 - remainder
    }
}

fn record_stride(variables: &[netcdf_reader::NcVariable]) -> Result<u64> {
    let record_variables = variables
        .iter()
        .filter(|variable| variable.is_record_var)
        .collect::<Vec<_>>();
    if record_variables.len() == 1 {
        return Ok(record_variables[0].record_size);
    }
    record_variables.iter().try_fold(0_u64, |stride, variable| {
        stride
            .checked_add(padded_record_size(variable.record_size))
            .ok_or_else(|| NcvError::InvalidSlice("NetCDF-3 record stride overflows".into()))
    })
}

fn decode_numeric(raw: &[u8], dtype: &NcType, count: usize) -> Result<Vec<f64>> {
    let size = dtype.size().map_err(|error| NcvError::Adapter {
        path: PathBuf::new(),
        reason: error.to_string(),
    })?;
    if raw.len() < count.saturating_mul(size) {
        return Err(NcvError::Adapter {
            path: PathBuf::new(),
            reason: "NetCDF-3 record payload is truncated".into(),
        });
    }
    let mut values = Vec::with_capacity(count);
    for chunk in raw[..count * size].chunks_exact(size) {
        let value = match dtype {
            NcType::Byte => f64::from(chunk[0] as i8),
            NcType::UByte => f64::from(chunk[0]),
            NcType::Short => f64::from(i16::from_be_bytes([chunk[0], chunk[1]])),
            NcType::UShort => f64::from(u16::from_be_bytes([chunk[0], chunk[1]])),
            NcType::Int => f64::from(i32::from_be_bytes(chunk.try_into().unwrap())),
            NcType::UInt => u32::from_be_bytes(chunk.try_into().unwrap()) as f64,
            NcType::Float => f64::from(f32::from_bits(u32::from_be_bytes(
                chunk.try_into().unwrap(),
            ))),
            NcType::Double => f64::from_bits(u64::from_be_bytes(chunk.try_into().unwrap())),
            NcType::Int64 => i64::from_be_bytes(chunk.try_into().unwrap()) as f64,
            NcType::UInt64 => u64::from_be_bytes(chunk.try_into().unwrap()) as f64,
            _ => {
                return Err(NcvError::UnsupportedVariable {
                    variable: "record variable".into(),
                    reason: format!("non-numeric NetCDF-3 type {dtype:?}"),
                });
            }
        };
        values.push(value);
    }
    Ok(values)
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

fn metadata(path: &Path, file: &NcFile) -> Result<DatasetMetadata> {
    let dimensions = file
        .dimensions()
        .map_err(|error| NcvError::InvalidDataset {
            path: path.to_path_buf(),
            reason: error.to_string(),
        })?
        .iter()
        .map(|dimension| Dimension {
            name: dimension.name.clone(),
            length: usize::try_from(dimension.size).unwrap_or(usize::MAX),
            role: role_for_name(&dimension.name),
        })
        .collect::<Vec<_>>();
    let variables = file
        .variables()
        .map_err(|error| NcvError::InvalidDataset {
            path: path.to_path_buf(),
            reason: error.to_string(),
        })?
        .iter()
        .map(|variable| Variable {
            name: variable.name.clone(),
            dimensions: variable
                .dimensions
                .iter()
                .map(|dimension| dimension.name.clone())
                .collect(),
            numeric: is_numeric_type(&variable.dtype),
            units: attribute_text(variable.attributes.as_slice(), "units"),
            long_name: attribute_text(variable.attributes.as_slice(), "long_name"),
            standard_name: attribute_text(variable.attributes.as_slice(), "standard_name"),
        })
        .collect();
    Ok(DatasetMetadata {
        path: path.display().to_string(),
        format: DatasetFormat::NetCdf3,
        dimensions,
        variables,
    })
}

fn attribute_text(attributes: &[netcdf_reader::NcAttribute], name: &str) -> Option<String> {
    attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .and_then(|attribute| match &attribute.value {
            NcAttrValue::Chars(value) => Some(value.clone()),
            NcAttrValue::Strings(values) => values.first().cloned(),
            _ => None,
        })
}

fn is_numeric_type(dtype: &NcType) -> bool {
    matches!(
        dtype,
        NcType::Byte
            | NcType::Short
            | NcType::Int
            | NcType::Float
            | NcType::Double
            | NcType::UByte
            | NcType::UShort
            | NcType::UInt
            | NcType::Int64
            | NcType::UInt64
    )
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
