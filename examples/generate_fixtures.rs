use std::{env, fs, path::PathBuf};

use oxinetcdf::{NcFileWriter, NcType, VarOrGroup};

fn write_field(path: PathBuf, rows: usize, cols: usize, curvilinear: bool) {
    let mut writer = NcFileWriter::new();
    let lat = writer.def_dim("lat", rows).unwrap();
    let lon = writer.def_dim("lon", cols).unwrap();
    let field = writer
        .def_var("temperature", &[lat, lon], NcType::Float64)
        .unwrap();
    writer
        .put_att_str(VarOrGroup::Var(field), "units", "K")
        .unwrap();
    let values = (0..rows * cols)
        .map(|index| 250.0 + index as f64 * 0.25)
        .collect::<Vec<_>>();
    writer.put_var_f64(field, &values).unwrap();
    if curvilinear {
        let lat2 = writer
            .def_var("lat2d", &[lat, lon], NcType::Float64)
            .unwrap();
        let lon2 = writer
            .def_var("lon2d", &[lat, lon], NcType::Float64)
            .unwrap();
        writer
            .put_var_f64(
                lat2,
                &(0..rows * cols)
                    .map(|index| index as f64 / cols as f64)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        writer
            .put_var_f64(
                lon2,
                &(0..rows * cols)
                    .map(|index| index as f64 % cols as f64)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
    }
    writer.close(path).unwrap();
}

fn main() {
    let output = PathBuf::from(
        env::args()
            .nth(1)
            .unwrap_or_else(|| "tests/fixtures".into()),
    );
    fs::create_dir_all(&output).unwrap();
    write_field(output.join("regular.nc4"), 4, 5, false);
    write_field(output.join("packed-fill.nc4"), 4, 5, false);
    write_field(output.join("curvilinear.nc4"), 4, 5, true);
    fs::write(output.join("netcdf3-unsupported.nc"), b"CDF\x01legacy").unwrap();
    fs::write(output.join("corrupt.nc"), b"not a dataset").unwrap();
    fs::write(output.join("arbitrary.h5"), b"\x89HDF\r\n\x1a\nnot-netcdf").unwrap();
}
