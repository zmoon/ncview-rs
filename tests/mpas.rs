use std::path::Path;

use ncview_rs::data::{
    self,
    slice::{Bounds, SliceRequest},
};
use oxinetcdf::{NcFileWriter, NcType};
use tempfile::tempdir;

#[test]
fn opens_committed_netcdf3_mpas_fixture_and_resamples() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mpas-small.nc");
    let source = data::open(&path).expect("open committed MPAS NetCDF-3 fixture");

    assert_eq!(source.metadata().format, data::DatasetFormat::NetCdf3);
    let bounds = Bounds::new(0, 3, 0, 4).unwrap();
    let slice = source
        .read_slice(&SliceRequest {
            variable: "temperature".into(),
            time: 0,
            depth: 0,
            bounds,
        })
        .expect("resample the MPAS cell field");

    assert_eq!(slice.values.dim(), (3, 4));
    assert_eq!(slice.source_bounds, bounds);
    assert!(slice.values.iter().all(|value| value.is_finite()));
    assert!(slice.coordinates.is_some());
}

#[test]
#[ignore = "requires the downloaded MPAS x1.2562 mesh; set MPAS_MESH"]
fn opens_real_mpas_mesh_and_resamples_coordinate_field() {
    let path = std::env::var("MPAS_MESH").expect("set MPAS_MESH to x1.2562.grid.nc");
    let mesh_source = data::open(&path).expect("open MPAS mesh");
    let metadata = mesh_source.metadata();

    let n_cells = metadata
        .dimensions
        .iter()
        .find(|dimension| dimension.name == "nCells")
        .map(|dimension| dimension.length)
        .expect("MPAS mesh nCells dimension");
    assert!(n_cells > 4);
    assert!(
        metadata
            .variables
            .iter()
            .any(|variable| variable.name == "latCell")
    );
    assert!(
        metadata
            .variables
            .iter()
            .any(|variable| variable.name == "lonCell")
    );

    let mesh_slice = mesh_source
        .read_slice(&SliceRequest {
            variable: "latCell".into(),
            time: 0,
            depth: 0,
            bounds: Bounds::new(0, 4, 0, 1).unwrap(),
        })
        .expect("read a mesh variable from the native NetCDF-3 source");
    assert!(mesh_slice.values.iter().all(|value| value.is_finite()));

    let temp_dir = tempdir().unwrap();
    let field_path = temp_dir.path().join("field.nc4");
    let mut writer = NcFileWriter::new();
    let n_cells_dimension = writer.def_dim("nCells", n_cells).unwrap();
    let temperature = writer
        .def_var("temperature", &[n_cells_dimension], NcType::Float64)
        .unwrap();
    let values = (0..n_cells)
        .map(|index| 280.0 + index as f64 * 0.01)
        .collect::<Vec<_>>();
    writer.put_var_f64(temperature, &values).unwrap();
    writer.close(&field_path).unwrap();

    let source = data::open_with_grid(&field_path, Some(Path::new(&path)))
        .expect("open MPAS field with external mesh coordinates");
    assert!(
        source
            .metadata()
            .dimensions
            .iter()
            .any(|dimension| { dimension.name == "nCells" && dimension.length > 4 })
    );

    let bounds = Bounds::new(0, 4, 0, 6).unwrap();
    let slice = source
        .read_slice(&SliceRequest {
            variable: "temperature".into(),
            time: 0,
            depth: 0,
            bounds,
        })
        .expect("resample latCell onto the requested window");

    assert_eq!(slice.values.dim(), (4, 6));
    assert_eq!(slice.source_bounds, bounds);
    let coordinates = slice.coordinates.expect("synthetic grid coordinates");
    assert_eq!(coordinates.latitude_axis.as_ref().unwrap().len(), 4);
    assert_eq!(coordinates.longitude_axis.as_ref().unwrap().len(), 6);
    assert!(slice.values.iter().all(|value| value.is_finite()));
    assert!(slice.values.iter().any(|value| *value != 0.0));
}
