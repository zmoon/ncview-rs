use std::{
    env, fs, io,
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
    process::ExitCode,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use chrono::{DateTime, SecondsFormat, TimeDelta, Utc};
use clap::{CommandFactory, Parser, Subcommand};
use crossterm::{
    event, execute,
    terminal::{LeaveAlternateScreen, disable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend, layout::Rect};

use ncview_rs::{
    analysis::mapping::screen_to_source_with_row_flip,
    app::{
        AppState, AxisField, ColorScaleScope, Command, Generation, LimitField, Overlay, PlotSeries,
        TimelinePoint,
    },
    data::{
        self, AxisRole, DatasetFormat, DatasetMetadata, Variable,
        grib2_manifest::{self, ManifestFormat},
        slice::{Bounds, Slice2D, SliceRequest},
        virtual_dataset::VirtualDatasetManifest,
    },
    events::input,
    render::protocol::GraphicsRenderer,
    storage::operation::OperationPhase,
    ui::{dashboard, layout as dashboard_layout, status},
};

#[derive(Debug, Parser)]
#[command(
    name = "ncv",
    version,
    about = "Terminal-native NetCDF and GRIB2 scientific data viewer"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<CliCommand>,
    /// One or more NetCDF-3, NetCDF-4, or GRIB2 datasets to inspect. Shell globs are supported.
    #[arg(value_name = "DATASET", num_args = 0..)]
    dataset: Vec<String>,
    /// MPAS mesh/coordinate file supplying latCell/lonCell or latVertex/lonVertex.
    #[arg(long, value_name = "GRID")]
    grid: Option<String>,
}

#[derive(Debug, Subcommand)]
enum CliCommand {
    /// Create a Kerchunk-compatible GRIB2 reference manifest from a `.idx` sidecar.
    Manifest {
        /// Output profile. `virtualizarr` emits a VirtualiZarr-consumable Kerchunk profile.
        #[arg(long, value_parser = ["kerchunk", "virtualizarr"])]
        format: String,
        /// GRIB2 source object.
        #[arg(long)]
        input: String,
        /// Matching NOAA-style `.idx` sidecar.
        #[arg(long)]
        idx: String,
        /// Manifest destination JSON.
        #[arg(long)]
        output: String,
        /// URI to place in byte-range references instead of the local source path.
        #[arg(long)]
        source_uri: Option<String>,
        /// Treat warnings and mismatches as errors.
        #[arg(long)]
        strict: bool,
    },
}

fn setup_panic_hook() {
    let _ = color_eyre::install();
    human_panic::setup_panic!();
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let _ = disable_raw_mode();
        let mut stdout = io::stdout();
        let _ = execute!(
            stdout,
            crossterm::cursor::Show,
            crossterm::event::DisableMouseCapture,
            LeaveAlternateScreen
        );
        let _ = io::Write::flush(&mut stdout);
        prev_hook(panic_info);
    }));
}

fn main() -> ExitCode {
    setup_panic_hook();
    ncview_rs::render::configure_thread_pool();
    let cli = Cli::parse();
    if let Some(CliCommand::Manifest {
        format,
        input,
        idx,
        output,
        source_uri,
        strict,
    }) = cli.command
    {
        let manifest_format = match format.as_str() {
            "kerchunk" => ManifestFormat::Kerchunk,
            "virtualizarr" => ManifestFormat::Virtualizarr,
            _ => unreachable!("clap validates manifest format"),
        };
        return match grib2_manifest::write_manifest(
            Path::new(&input),
            Path::new(&idx),
            Path::new(&output),
            manifest_format,
            source_uri.as_deref(),
            strict,
        ) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("ncv manifest: {error}");
                ExitCode::from(2)
            }
        };
    }
    if cli.dataset.is_empty() {
        let mut command = Cli::command();
        let _ = command.print_help();
        println!();
        return ExitCode::SUCCESS;
    }
    for dataset in &cli.dataset {
        if let Err(error) = ncview_rs::storage::location::SourceLocation::parse(dataset) {
            eprintln!("ncv: {dataset}: {error}");
            return ExitCode::from(2);
        }
    }
    if let Err(error) = run(&cli.dataset, cli.grid.as_deref()) {
        eprintln!("ncv: {error}");
        return ExitCode::from(2);
    }
    ExitCode::SUCCESS
}

fn run(datasets: &[String], grid: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let datasets = datasets.to_vec();
    let grid = grid.map(str::to_owned);
    let (stdout_tx, stdout_rx) = std::sync::mpsc::channel::<Vec<u8>>();
    std::thread::spawn(move || {
        use std::io::Write;
        while let Ok(bytes) = stdout_rx.recv() {
            // Do not retain this lock while waiting for the next frame:
            // TerminalSession needs stdout to enter and leave the alternate
            // screen, and holding it here deadlocks startup.
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            let _ = handle.write_all(&bytes);
            let _ = handle.flush();
        }
    });

    struct ChannelWriter {
        tx: std::sync::mpsc::Sender<Vec<u8>>,
        buffer: Vec<u8>,
    }

    impl std::io::Write for ChannelWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.buffer.extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            if !self.buffer.is_empty() {
                let bytes = std::mem::take(&mut self.buffer);
                let _ = self.tx.send(bytes);
            }
            Ok(())
        }
    }

    let channel_writer = ChannelWriter {
        tx: stdout_tx,
        buffer: Vec::new(),
    };

    let mut session = ncview_rs::events::terminal::TerminalSession::enter()?;
    let backend = CrosstermBackend::new(channel_writer);
    let mut terminal = Terminal::new(backend)?;
    let mut graphics = GraphicsRenderer::probe();
    let mut chart_graphics = graphics.secondary();

    let lazy_remote_grib_collection = lazy_remote_grib_collection(&datasets);
    let (load_tx, load_rx) = std::sync::mpsc::channel();
    let cancelled = Arc::new(AtomicBool::new(false));
    for (index, dataset) in datasets.iter().cloned().enumerate() {
        if lazy_remote_grib_collection && index > 0 {
            continue;
        }
        let load_tx = load_tx.clone();
        let worker_cancelled = Arc::clone(&cancelled);
        let grid_for_worker = grid.clone();
        std::thread::spawn(move || {
            if worker_cancelled.load(Ordering::Relaxed) {
                return;
            }
            if load_tx
                .send(SourceLoadMessage::Started(index, OperationPhase::Queued))
                .is_err()
            {
                return;
            }
            let progress_tx = load_tx.clone();
            let progress_cancelled = Arc::clone(&worker_cancelled);
            let result = catch_unwind(AssertUnwindSafe(|| {
                if let Some(grid_path) = grid_for_worker.as_deref() {
                    data::open_location_with_grid(&dataset, Some(grid_path))
                } else {
                    data::open_location_with_progress(&dataset, &|message| {
                        if progress_cancelled.load(Ordering::Relaxed) {
                            return false;
                        }
                        progress_tx
                            .send(SourceLoadMessage::Progress(
                                index,
                                loading_phase(message),
                                message.to_owned(),
                            ))
                            .is_ok()
                    })
                }
                .and_then(|source| {
                    if progress_cancelled.load(Ordering::Relaxed) {
                        return Err(ncview_rs::error::NcvError::WorkerStopped);
                    }
                    Ok(source)
                })
            }))
            .map_err(|_| format!("{dataset}: loader panicked"))
            .and_then(|result| result.map_err(|error| format!("{dataset}: {error}")));
            let _ = load_tx.send(SourceLoadMessage::Finished(index, result));
        });
    }
    drop(load_tx);

    let mut pending_sources: Vec<Option<Arc<dyn data::DataSource>>> =
        (0..datasets.len()).map(|_| None).collect();
    let mut loaded = 0usize;
    let mut finished = 0usize;
    let mut load_errors = Vec::new();
    let mut current_dataset = None;
    let mut loading_status = None;
    loop {
        while let Ok(message) = load_rx.try_recv() {
            match message {
                SourceLoadMessage::Started(index, phase) => {
                    current_dataset = datasets.get(index).cloned();
                    loading_status = Some(phase_label(phase).to_owned());
                }
                SourceLoadMessage::Progress(index, phase, message) => {
                    current_dataset = datasets.get(index).cloned();
                    loading_status = Some(format!("{}: {message}", phase_label(phase)));
                }
                SourceLoadMessage::Finished(index, result) => match result {
                    Ok(source) => {
                        pending_sources[index] = Some(Arc::from(source));
                        loaded += 1;
                        finished += 1;
                        loading_status = Some("metadata ready".to_owned());
                    }
                    Err(error) => {
                        finished += 1;
                        load_errors.push((index, error));
                        loading_status =
                            Some("input failed; continuing with other sources".to_owned());
                    }
                },
            }
        }
        terminal.draw(|frame| {
            status::render_loading(
                frame,
                frame.area(),
                &datasets,
                loaded,
                current_dataset.as_deref(),
                loading_status.as_deref(),
            );
        })?;
        let metadata_loading_complete =
            finished == datasets.len() || (lazy_remote_grib_collection && finished > 0);
        if metadata_loading_complete {
            if loaded > 0 {
                break;
            }
            cancelled.store(true, Ordering::Relaxed);
            let details = load_errors
                .into_iter()
                .map(|(index, error)| format!("{}: {error}", datasets[index]))
                .collect::<Vec<_>>()
                .join("; ");
            return Err(details.into());
        }
        if event::poll(Duration::from_millis(50))?
            && let event::Event::Key(key) = event::read()?
            && matches!(key.code, crossterm::event::KeyCode::Char('q'))
        {
            cancelled.store(true, Ordering::Relaxed);
            return Ok(());
        }
    }

    let first_loaded = pending_sources
        .iter()
        .position(Option::is_some)
        .ok_or("no dataset finished loading")?;
    if lazy_remote_grib_collection {
        let template = pending_sources[first_loaded]
            .as_ref()
            .ok_or("initial remote GRIB2 source disappeared")?;
        let template_metadata = template.metadata().clone();
        let first_variable = template
            .metadata()
            .variables
            .iter()
            .find(|variable| variable.numeric && plottable_variable(variable))
            .map(|variable| variable.name.clone());
        let base_label = first_variable
            .as_deref()
            .and_then(|variable| template.time_label_for_variable(variable, 0));
        let base_hour = datasets
            .get(first_loaded)
            .and_then(|dataset| forecast_hour(dataset))
            .unwrap_or(0);
        for (index, dataset) in datasets.iter().enumerate() {
            if index == first_loaded {
                continue;
            }
            pending_sources[index] = Some(Arc::new(LazyRemoteGribSource::new(
                dataset,
                &template_metadata,
                base_label.as_deref(),
                base_hour,
                forecast_hour(dataset).unwrap_or(base_hour),
            )));
        }
        loaded = datasets.len();
        finished = datasets.len();
    }
    let mut sources = pending_sources
        .into_iter()
        .enumerate()
        .map(|(index, source)| {
            source.unwrap_or_else(|| placeholder_source(datasets[index].clone()))
        })
        .collect::<Vec<Arc<dyn data::DataSource>>>();
    let source_refs = sources
        .iter()
        .map(|source| source.as_ref())
        .collect::<Vec<_>>();
    let mut manifest = VirtualDatasetManifest::from_sources(&source_refs);
    let mut active_file = earliest_source_index(&sources);
    let initial_source = sources[active_file].as_ref();
    let mut state = state_for_source(initial_source);
    state.view.collection_diagnostics = manifest.diagnostics().to_vec();
    state.view.collection_progress = Some((finished, datasets.len()));
    state.view.collection_diagnostics.extend(
        load_errors
            .iter()
            .map(|(index, error)| format!("{}: {error}", datasets[*index])),
    );
    if lazy_remote_grib_collection {
        state
            .view
            .collection_diagnostics
            .push("remote GRIB2 collection ready; later forecast files open on demand".into());
    }
    if finished < datasets.len() {
        state.view.collection_diagnostics.push(format!(
            "opening {}/{} sources; remaining inputs continue in the background",
            loaded,
            datasets.len()
        ));
    }
    let (slice_tx, slice_rx) = std::sync::mpsc::channel::<RemoteSliceMessage>();
    let mut slice_cancelled = Arc::new(AtomicBool::new(false));
    let (plot_tx, plot_rx) = std::sync::mpsc::channel::<RemotePlotMessage>();
    let mut plot_cancelled = Arc::new(AtomicBool::new(false));
    select_initial_variable(&mut state, initial_source.metadata());
    configure_timeline(&mut state, &sources, &manifest, active_file);
    terminal.draw(|frame| {
        status::render_loading(
            frame,
            frame.area(),
            &datasets,
            loaded,
            Some("preparing initial field"),
            None,
        );
    })?;
    load_selected(
        &mut state,
        &sources,
        active_file,
        &slice_tx,
        &mut slice_cancelled,
    );
    if state.view.slice.is_none() {
        state.view.status = format!(
            "opened {} variable(s); no plottable data",
            state.variables.len()
        );
    }
    let mut dirty = true;
    let mut last_render = Instant::now();
    let frame_budget = Duration::from_millis(33); // ~30 FPS throttle max
    let mut last_playback_tick = Instant::now();

    loop {
        while let Ok(message) = load_rx.try_recv() {
            match message {
                SourceLoadMessage::Started(_, _) | SourceLoadMessage::Progress(_, _, _) => {}
                SourceLoadMessage::Finished(index, result) => match result {
                    Ok(source) => {
                        sources[index] = Arc::from(source);
                        loaded += 1;
                        finished += 1;
                        state.view.collection_progress = Some((finished, datasets.len()));
                        manifest = VirtualDatasetManifest::from_sources(
                            &sources
                                .iter()
                                .map(|source| source.as_ref())
                                .collect::<Vec<_>>(),
                        );
                        state.variables = collection_variables(&sources);
                        state.view.collection_diagnostics = manifest.diagnostics().to_vec();
                        if loaded < datasets.len() {
                            state.view.collection_diagnostics.push(format!(
                                "opening {}/{} sources in background",
                                loaded,
                                datasets.len()
                            ));
                        }
                        if finished == datasets.len() {
                            state.view.status = format!(
                                "opened {}/{} sources; collection loading complete",
                                loaded,
                                datasets.len()
                            );
                        }
                        configure_timeline(&mut state, &sources, &manifest, active_file);
                    }
                    Err(error) => {
                        finished += 1;
                        state.view.collection_progress = Some((finished, datasets.len()));
                        state
                            .view
                            .collection_diagnostics
                            .push(format!("{}: {error}", datasets[index]));
                        if finished == datasets.len() {
                            state.view.status = format!(
                                "opened {}/{} sources; collection loading complete",
                                loaded,
                                datasets.len()
                            );
                        }
                    }
                },
            }
            dirty = true;
        }
        while let Ok(message) = slice_rx.try_recv() {
            apply_remote_slice(&mut state, &sources, message);
            dirty = true;
        }
        while let Ok(message) = plot_rx.try_recv() {
            apply_remote_plot(&mut state, message);
            dirty = true;
        }
        if state.view.playing
            && last_playback_tick.elapsed()
                >= Duration::from_secs_f32(1.0 / state.view.playback_speed)
        {
            state.reduce(Command::TickPlayback);
            load_selected(
                &mut state,
                &sources,
                active_file,
                &slice_tx,
                &mut slice_cancelled,
            );
            last_playback_tick = Instant::now();
            dirty = true;
        }
        // A timeline frame is backed by a source, not just by a local time
        // coordinate. Keep the active source synchronized before rendering
        // and before handling the next input so a frame change can actually
        // switch the remote object being read.
        if let Some(point) = state.view.timeline.get(state.view.time_index) {
            active_file = point.source_index;
        }
        let source = sources[active_file].as_ref();

        if (dirty || graphics.has_pending_image() || chart_graphics.has_pending_image())
            && last_render.elapsed() >= frame_budget
        {
            terminal.draw(|frame| {
                dashboard::render_with_search_and_image(
                    frame,
                    frame.area(),
                    &state.view,
                    &display_dataset_name(&datasets[active_file], active_file, datasets.len()),
                    source.metadata(),
                    &state.variable_query,
                    state.view.variable_search_active,
                    Some(&mut graphics),
                    Some(&mut chart_graphics),
                )
            })?;
            last_render = Instant::now();
            dirty = false;
        }

        // Compute poll timeout: idle if not playing, else remaining time to next playback tick/render
        let poll_timeout = if state.view.playing {
            let playback_interval = Duration::from_secs_f32(1.0 / state.view.playback_speed);
            let elapsed = last_playback_tick.elapsed();
            playback_interval
                .saturating_sub(elapsed)
                .min(Duration::from_millis(33))
        } else if dirty || graphics.has_pending_image() || chart_graphics.has_pending_image() {
            frame_budget.saturating_sub(last_render.elapsed())
        } else {
            Duration::from_millis(100)
        };

        if event::poll(poll_timeout)?
            && let Some(command) =
                input::command_from_event_with_mode(event::read()?, state.view.input_mode())
        {
            dirty = true;
            let size = terminal.size()?;
            let command = translate_mouse(
                command,
                Rect::new(0, 0, size.width, size.height),
                source.metadata(),
                &state.view,
                &state.variable_query,
                Some(&graphics),
            );
            if matches!(command, Command::Quit)
                && state.view.overlay.is_none()
                && !state.view.variable_search_active
                && !state.view.help_visible
            {
                break;
            }
            if handle_command(
                command,
                &mut state,
                &sources,
                &datasets,
                &mut active_file,
                &manifest,
                finished,
                &slice_tx,
                &mut slice_cancelled,
                &plot_tx,
                &mut plot_cancelled,
                &mut graphics,
            ) {
                continue;
            }
        }
    }
    cancelled.store(true, Ordering::Release);
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    session.restore()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle_command(
    command: Command,
    state: &mut AppState,
    sources: &[Arc<dyn data::DataSource>],
    datasets: &[String],
    active_file: &mut usize,
    manifest: &VirtualDatasetManifest,
    finished: usize,
    slice_tx: &std::sync::mpsc::Sender<RemoteSliceMessage>,
    slice_cancelled: &mut Arc<AtomicBool>,
    plot_tx: &std::sync::mpsc::Sender<RemotePlotMessage>,
    plot_cancelled: &mut Arc<AtomicBool>,
    graphics: &mut GraphicsRenderer,
) -> bool {
    let file_delta = match command {
        Command::PreviousFile => Some(-1isize),
        Command::NextFile => Some(1isize),
        _ => None,
    };
    if let Some(delta) = file_delta {
        let palette = state.view.palette.clone();
        let scale_mode = state.view.scale_mode;
        let color_scale_scope = state.view.color_scale_scope;
        let grid_mode = state.view.grid_mode;
        let show_land_borders = state.view.show_land_borders;
        let playback_speed = state.view.playback_speed;
        let plot_generation = state.view.plot_generation;

        *active_file = bounded_file_index(*active_file, delta, sources.len());
        let source = sources[*active_file].as_ref();
        *state = state_for_source(source);
        state.view.collection_diagnostics = manifest.diagnostics().to_vec();
        state.view.collection_progress = Some((finished, datasets.len()));
        state.view.palette = palette;
        state.view.scale_mode = scale_mode;
        state.view.color_scale_scope = color_scale_scope;
        state.view.grid_mode = grid_mode;
        state.view.show_land_borders = show_land_borders;
        state.view.playback_speed = playback_speed;
        state.view.plot_generation = plot_generation;

        select_initial_variable(state, source.metadata());
        configure_timeline(state, sources, manifest, *active_file);
        load_selected(state, sources, *active_file, slice_tx, slice_cancelled);
        state.view.status = format!(
            "opened file {}/{}: {}",
            *active_file + 1,
            datasets.len(),
            datasets[*active_file]
        );
        return true;
    }

    let source = sources[*active_file].as_ref();
    let dataset = &datasets[*active_file];
    let activate_point = matches!(command, Command::ActivatePoint);
    let cycle_image_filter = matches!(command, Command::CycleImageFilter);
    let export_current = matches!(command, Command::ExportCurrent);
    let refresh_time_series = activate_point
        || matches!(command, Command::OpenPlot)
        || matches!(
            command,
            Command::MoveTime(_)
                | Command::SetTime(_)
                | Command::MoveDepth(_)
                | Command::SetDepth(_)
                | Command::ApplyDepthCursor
                | Command::CyclePlotAxis(_)
                | Command::SetPlotKind(_)
                | Command::TogglePointSelection
                | Command::SelectVariable(_)
                | Command::SelectVariableAt(_)
                | Command::SubmitVariableSearch
                | Command::ExecuteCommandPalette
                | Command::SetAxes { .. }
        );
    let reload = matches!(
        command,
        Command::SelectVariable(_)
            | Command::SelectVariableAt(_)
            | Command::MoveTime(_)
            | Command::SetTime(_)
            | Command::MoveDepth(_)
            | Command::SetDepth(_)
            | Command::ApplyDepthCursor
            | Command::TickPlayback
            | Command::SubmitVariableSearch
            | Command::ExecuteCommandPalette
            | Command::Zoom(_)
            | Command::ResetZoom
            | Command::Pan { .. }
            | Command::SetAxes { .. }
            | Command::AutomaticLimits
            | Command::ToggleColorScaleScope
            | Command::ToggleScale
    );
    let axis_submit =
        matches!(command, Command::ActivatePoint) && state.view.overlay == Some(Overlay::Axis);
    let reconfigure_timeline = matches!(
        command,
        Command::SelectVariable(_)
            | Command::SelectVariableAt(_)
            | Command::SubmitVariableSearch
            | Command::ExecuteCommandPalette
            | Command::SetAxes { .. }
    ) || axis_submit;
    let point_target = match &command {
        Command::HoverPoint { row, col, .. } | Command::SelectPoint { row, col } => {
            Some((*row, *col))
        }
        Command::TogglePointSelection => state
            .view
            .hover_point
            .as_ref()
            .map(|point| (point.row, point.col)),
        Command::ActivatePoint => state.view.selected_point,
        _ => None,
    };
    let _ = state.reduce(command);
    if cycle_image_filter {
        if graphics.cycle_filter() {
            state.view.status = format!("image interpolation: {}", graphics.filter_label());
        } else {
            state.view.status =
                "scientific rendering locked; set NCVIEW_SCIENTIFIC_RENDERING=0 to enable interpolation"
                    .into();
        }
    }
    if export_current {
        match export_current_slice(state, dataset, source.metadata()) {
            Ok(path) => state.view.status = format!("exported {}", path.display()),
            Err(error) => state.view.status = format!("export failed: {error}"),
        }
    }
    if let Some((row, col)) = point_target
        && let Some(variable) = state.view.selected_variable.as_deref()
    {
        let coordinates = source.point_coordinates(variable, row, col);
        if let Some(point) = state.view.hover_point.as_mut()
            && point.row == row
            && point.col == col
        {
            point.latitude = coordinates.latitude;
            point.longitude = coordinates.longitude;
        }
        if state.view.selected_point == Some((row, col)) {
            state.view.selected_coordinates = coordinates;
        }
    }
    if reload || axis_submit {
        if reconfigure_timeline {
            configure_timeline(state, sources, manifest, *active_file);
        }
        load_selected(state, sources, *active_file, slice_tx, slice_cancelled);
    }
    if refresh_time_series
        && matches!(
            state.view.overlay,
            Some(Overlay::TimeSeries | Overlay::Plot)
        )
    {
        load_time_series(state, sources, *active_file, plot_tx, plot_cancelled);
    }
    if let Some(point) = state.view.timeline.get(state.view.time_index) {
        *active_file = point.source_index;
    }
    false
}

enum SourceLoadMessage {
    Started(usize, OperationPhase),
    Progress(usize, OperationPhase, String),
    Finished(usize, Result<Box<dyn data::DataSource>, String>),
}

struct PlaceholderSource {
    metadata: DatasetMetadata,
}

fn placeholder_source(path: String) -> Arc<dyn data::DataSource> {
    Arc::new(PlaceholderSource {
        metadata: DatasetMetadata {
            path,
            format: DatasetFormat::NetCdf4,
            dimensions: Vec::new(),
            variables: Vec::new(),
        },
    })
}

impl data::DataSource for PlaceholderSource {
    fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }

    fn read_slice(&self, _request: &SliceRequest) -> ncview_rs::error::Result<Slice2D> {
        Err(ncview_rs::error::NcvError::InvalidDataset {
            path: PathBuf::from(&self.metadata.path),
            reason: "source metadata is still loading".into(),
        })
    }
}

fn collection_variables(sources: &[Arc<dyn data::DataSource>]) -> Vec<Variable> {
    let mut variables = Vec::new();
    for source in sources {
        for variable in source
            .metadata()
            .variables
            .iter()
            .filter(|variable| variable.numeric && plottable_variable(variable))
        {
            if !variables
                .iter()
                .any(|candidate: &Variable| candidate.name == variable.name)
            {
                variables.push(variable.clone());
            }
        }
    }
    variables
}

fn lazy_remote_grib_collection(datasets: &[String]) -> bool {
    let Some(first_key) = datasets
        .first()
        .and_then(|dataset| forecast_collection_key(dataset))
    else {
        return false;
    };
    datasets.len() > 1
        && datasets
            .iter()
            .all(|dataset| forecast_collection_key(dataset).is_some_and(|key| key == first_key))
}

fn forecast_collection_key(dataset: &str) -> Option<String> {
    let location = ncview_rs::storage::location::SourceLocation::parse(dataset).ok()?;
    if !location.is_remote()
        || !location
            .object_key()
            .rsplit_once('.')
            .is_some_and(|(_, extension)| {
                matches!(
                    extension.to_ascii_lowercase().as_str(),
                    "grib" | "grib2" | "grb" | "grb2"
                )
            })
    {
        return None;
    }
    let marker = location.object_key().rfind(".f")?;
    let suffix = &location.object_key()[marker + 2..];
    if suffix.len() < 3
        || !suffix[..3]
            .chars()
            .all(|character| character.is_ascii_digit())
    {
        return None;
    }
    Some(format!(
        "{}{}",
        &location.object_key()[..marker],
        &suffix[3..]
    ))
}

fn forecast_hour(dataset: &str) -> Option<u32> {
    let marker = dataset.rfind(".f")? + 2;
    let digits = dataset[marker..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

struct LazyRemoteGribSource {
    path: String,
    metadata: DatasetMetadata,
    time_label: Option<String>,
    source: std::sync::OnceLock<Arc<dyn data::DataSource>>,
}

impl LazyRemoteGribSource {
    fn new(
        path: &str,
        template: &DatasetMetadata,
        base_label: Option<&str>,
        base_hour: u32,
        hour: u32,
    ) -> Self {
        let mut metadata = template.clone();
        metadata.path = path.to_owned();
        Self {
            path: path.to_owned(),
            metadata,
            time_label: valid_time_from_forecast_hour(base_label, base_hour, hour)
                .or_else(|| Some(format!("forecast f{hour:03}"))),
            source: std::sync::OnceLock::new(),
        }
    }

    fn source(&self) -> Result<Arc<dyn data::DataSource>, String> {
        if let Some(source) = self.source.get() {
            return Ok(Arc::clone(source));
        }
        let source = data::open_location(&self.path).map_err(|error| error.to_string())?;
        let source: Arc<dyn data::DataSource> = Arc::from(source);
        let _ = self.source.set(Arc::clone(&source));
        Ok(self
            .source
            .get()
            .map_or(source, |cached| Arc::clone(cached)))
    }

    fn request_for_source(
        &self,
        source: &dyn data::DataSource,
        request: &SliceRequest,
    ) -> Result<SliceRequest, String> {
        if source
            .metadata()
            .variables
            .iter()
            .any(|variable| variable.name == request.variable)
        {
            return Ok(request.clone());
        }
        let ordinal = self
            .metadata
            .variables
            .iter()
            .position(|variable| variable.name == request.variable)
            .ok_or_else(|| {
                format!(
                    "variable {} is not present in the collection",
                    request.variable
                )
            })?;
        let actual_variable = source
            .metadata()
            .variables
            .get(ordinal)
            .filter(|variable| variable.numeric && plottable_variable(variable))
            .map(|variable| variable.name.clone())
            .ok_or_else(|| {
                format!(
                    "variable {} is not present in the remote object",
                    request.variable
                )
            })?;
        let mut request = request.clone();
        request.variable = actual_variable;
        Ok(request)
    }
}

fn valid_time_from_forecast_hour(
    base_label: Option<&str>,
    base_hour: u32,
    hour: u32,
) -> Option<String> {
    let base = DateTime::parse_from_rfc3339(base_label?)
        .ok()?
        .with_timezone(&Utc);
    let offset = i64::from(hour).checked_sub(i64::from(base_hour))?;
    base.checked_add_signed(TimeDelta::try_hours(offset)?)
        .map(|time| time.to_rfc3339_opts(SecondsFormat::Secs, true))
}

impl data::DataSource for LazyRemoteGribSource {
    fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }

    fn is_remote(&self) -> bool {
        true
    }

    fn read_slice(&self, request: &SliceRequest) -> ncview_rs::error::Result<Slice2D> {
        let source = self
            .source()
            .map_err(|error| ncview_rs::error::NcvError::InvalidDataset {
                path: self.path.clone().into(),
                reason: format!("lazy remote GRIB2 open failed: {error}"),
            })?;
        let request = self
            .request_for_source(source.as_ref(), request)
            .map_err(|reason| ncview_rs::error::NcvError::InvalidDataset {
                path: self.path.clone().into(),
                reason,
            })?;
        source.read_slice(&request)
    }

    fn read_slice_on_axes(
        &self,
        request: &SliceRequest,
        row_axis: Option<&str>,
        col_axis: Option<&str>,
        fixed_axes: &[(String, usize)],
    ) -> ncview_rs::error::Result<Slice2D> {
        let source = self
            .source()
            .map_err(|error| ncview_rs::error::NcvError::InvalidDataset {
                path: self.path.clone().into(),
                reason: format!("lazy remote GRIB2 open failed: {error}"),
            })?;
        let request = self
            .request_for_source(source.as_ref(), request)
            .map_err(|reason| ncview_rs::error::NcvError::InvalidDataset {
                path: self.path.clone().into(),
                reason,
            })?;
        source.read_slice_on_axes(&request, row_axis, col_axis, fixed_axes)
    }

    fn read_slice_on_axes_cancellable(
        &self,
        request: &SliceRequest,
        row_axis: Option<&str>,
        col_axis: Option<&str>,
        fixed_axes: &[(String, usize)],
        cancelled: Arc<AtomicBool>,
    ) -> ncview_rs::error::Result<Slice2D> {
        let source = self
            .source()
            .map_err(|error| ncview_rs::error::NcvError::InvalidDataset {
                path: self.path.clone().into(),
                reason: format!("lazy remote GRIB2 open failed: {error}"),
            })?;
        let request = self
            .request_for_source(source.as_ref(), request)
            .map_err(|reason| ncview_rs::error::NcvError::InvalidDataset {
                path: self.path.clone().into(),
                reason,
            })?;
        source.read_slice_on_axes_cancellable(&request, row_axis, col_axis, fixed_axes, cancelled)
    }

    fn time_label(&self, _index: usize) -> Option<String> {
        self.time_label.clone()
    }

    fn time_label_for_variable(&self, _variable: &str, _index: usize) -> Option<String> {
        self.time_label.clone()
    }
}

fn loading_phase(message: &str) -> OperationPhase {
    let message = message.to_ascii_lowercase();
    if message.contains("connect") || message.contains("remote object") {
        OperationPhase::Head
    } else if message.contains("index") || message.contains("discover") {
        OperationPhase::Discovering
    } else if message.contains("fetch") {
        OperationPhase::Fetching
    } else if message.contains("decod") {
        OperationPhase::Decoding
    } else {
        OperationPhase::Queued
    }
}

fn phase_label(phase: OperationPhase) -> &'static str {
    match phase {
        OperationPhase::Queued => "queued",
        OperationPhase::Head => "opening object",
        OperationPhase::Discovering => "discovering metadata",
        OperationPhase::Fetching => "fetching data",
        OperationPhase::Decoding => "decoding data",
        OperationPhase::Rendering => "rendering",
        OperationPhase::Complete => "complete",
        OperationPhase::Failed => "failed",
        OperationPhase::Cancelled => "cancelled",
    }
}

struct RemoteSliceMessage {
    generation: Generation,
    source_index: usize,
    variable: String,
    full_view: bool,
    result: Result<(Slice2D, Option<(f64, f64)>), String>,
}

struct RemotePlotMessage {
    generation: Generation,
    result: Result<RemotePlotResult, String>,
}

struct RemotePlotResult {
    plot_series: Vec<PlotSeries>,
    time_series: Vec<(f64, f64)>,
    time_series_labels: Vec<String>,
    status: String,
}

fn state_for_source(source: &dyn data::DataSource) -> AppState {
    AppState {
        variables: source
            .metadata()
            .variables
            .iter()
            .filter(|variable| variable.numeric && plottable_variable(variable))
            .cloned()
            .collect(),
        ..AppState::default()
    }
}

fn display_dataset_name(path: &str, index: usize, total: usize) -> String {
    if total <= 1 {
        path.to_string()
    } else {
        format!("{path}  [{}/{}]", index + 1, total)
    }
}

fn bounded_file_index(index: usize, delta: isize, length: usize) -> usize {
    if length == 0 {
        return 0;
    }
    let next = index as isize + delta;
    next.clamp(0, length.saturating_sub(1) as isize) as usize
}

fn export_current_slice(
    state: &AppState,
    dataset: &str,
    metadata: &DatasetMetadata,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let slice = state
        .view
        .slice
        .as_ref()
        .ok_or("no slice is currently loaded")?;
    let variable_name = state
        .view
        .selected_variable
        .as_deref()
        .ok_or("no variable is currently selected")?;
    let metadata_variable = metadata
        .variables
        .iter()
        .find(|variable| variable.name == variable_name);
    let limits = state
        .view
        .limits
        .or_else(|| slice.statistics.map(|stats| (stats.min, stats.max)))
        .filter(|(min, max)| min.is_finite() && max.is_finite() && max > min)
        .ok_or("current slice has no finite color range")?;
    let raster = ncview_rs::render::raster::rgb_raster_with_options(
        slice,
        state.view.palette.clone(),
        Some(limits),
        state.view.filter_range,
        state.view.show_land_borders,
        state.view.scale_mode,
        None,
        state.view.selected_point,
    );
    let directory = env::var_os("NCVIEW_EXPORT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    fs::create_dir_all(&directory)?;
    let dataset_stem = Path::new(dataset)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(sanitize_filename)
        .unwrap_or_else(|| "ncv".into());
    let stem = sanitize_filename(variable_name);
    let stem = format!(
        "{dataset_stem}_{stem}_t{:04}_z{:04}",
        state.view.time_index, state.view.depth_index
    );
    let svg_path = directory.join(format!("{stem}.svg"));
    let png_path = directory.join(format!("{stem}.png"));
    let metadata_path = directory.join(format!("{stem}.json"));
    ncview_rs::export::write_slice_svg(
        &svg_path,
        &raster,
        &state.view.palette,
        limits,
        state.view.scale_mode,
        variable_name,
        metadata_variable.and_then(|variable| variable.units.as_deref()),
        metadata_variable.and_then(|variable| variable.long_name.as_deref()),
        metadata_variable.and_then(|variable| variable.standard_name.as_deref()),
        state.view.time_label.as_deref(),
        state.view.depth_index,
    )?;
    ncview_rs::export::write_slice_png(
        &png_path,
        &raster,
        &state.view.palette,
        limits,
        state.view.scale_mode,
        variable_name,
        metadata_variable.and_then(|variable| variable.units.as_deref()),
        metadata_variable.and_then(|variable| variable.long_name.as_deref()),
        metadata_variable.and_then(|variable| variable.standard_name.as_deref()),
        state.view.time_label.as_deref(),
        state.view.depth_index,
    )?;
    ncview_rs::export::write_slice_metadata_json(
        &metadata_path,
        limits,
        state.view.scale_mode,
        variable_name,
        metadata_variable.and_then(|variable| variable.units.as_deref()),
        metadata_variable.and_then(|variable| variable.long_name.as_deref()),
        metadata_variable.and_then(|variable| variable.standard_name.as_deref()),
        state.view.time_label.as_deref(),
        state.view.depth_index,
    )?;
    Ok(png_path)
}

fn sanitize_filename(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if sanitized.is_empty() {
        "slice".into()
    } else {
        sanitized
    }
}

/// Proportional depth index for a click at column `x` inside the level gauge,
/// matching the timeline's click-to-seek mapping (inner area, clamped).
fn depth_index_at(x: u16, area: Rect, depth_length: usize) -> usize {
    if depth_length <= 1 {
        return 0;
    }
    let start = area.x.saturating_add(1);
    let end = area.x.saturating_add(area.width.saturating_sub(2));
    let position = x.clamp(start, end).saturating_sub(start) as usize;
    let width = usize::from(end.saturating_sub(start).max(1));
    position.saturating_mul(depth_length.saturating_sub(1)) / width
}

fn translate_mouse(
    command: Command,
    area: Rect,
    metadata: &DatasetMetadata,
    view: &ncview_rs::app::ViewModel,
    variable_query: &str,
    graphics: Option<&GraphicsRenderer>,
) -> Command {
    match command {
        Command::BeginDrag { x, y, zoom } => translate_begin_drag(x, y, zoom, area, view, graphics),
        Command::UpdateDrag { x, y } => {
            if view.drag.is_none() {
                let areas = dashboard_layout::dashboard(area, view.depth_length > 1);
                if view.depth_length > 1 && areas.level.contains((x, y).into()) {
                    return Command::SetDepth(depth_index_at(x, areas.level, view.depth_length));
                }
            }
            translate_update_drag(x, y, view)
        }
        Command::MouseRelease { x, y } => {
            translate_mouse_release(x, y, area, metadata, view, variable_query, graphics)
        }
        Command::PointerScroll { x, y, delta } => {
            let areas = dashboard_layout::dashboard(area, view.depth_length > 1);
            if view.depth_length > 1 && areas.level.contains((x, y).into()) {
                return Command::MoveDepth(delta);
            }
            translate_scroll(x, y, delta, areas.sidebar, metadata, view, variable_query)
                .unwrap_or(Command::Pointer { x, y })
        }
        command => {
            translate_mouse_position(command, area, metadata, view, variable_query, graphics)
        }
    }
}

fn translate_begin_drag(
    x: u16,
    y: u16,
    zoom: bool,
    area: Rect,
    view: &ncview_rs::app::ViewModel,
    graphics: Option<&GraphicsRenderer>,
) -> Command {
    if view.help_visible || view.overlay.is_some() || view.variable_search_active {
        return Command::Pointer { x, y };
    }
    let areas = dashboard_layout::dashboard(area, view.depth_length > 1);
    if view.depth_length > 1 && areas.level.contains((x, y).into()) {
        return Command::SetDepth(depth_index_at(x, areas.level, view.depth_length));
    }
    let Some(canvas) = map_drawable(area, view, graphics) else {
        return Command::Pointer { x, y };
    };
    if canvas.contains((x, y).into()) {
        Command::BeginDrag { x, y, zoom }
    } else {
        Command::Pointer { x, y }
    }
}

fn translate_update_drag(x: u16, y: u16, view: &ncview_rs::app::ViewModel) -> Command {
    if view.drag.is_some() {
        Command::UpdateDrag { x, y }
    } else {
        Command::Pointer { x, y }
    }
}

fn translate_mouse_release(
    x: u16,
    y: u16,
    area: Rect,
    metadata: &DatasetMetadata,
    view: &ncview_rs::app::ViewModel,
    variable_query: &str,
    graphics: Option<&GraphicsRenderer>,
) -> Command {
    let Some(drag) = view.drag else {
        return translate_mouse_position(
            Command::MouseClick { x, y, right: false },
            area,
            metadata,
            view,
            variable_query,
            graphics,
        );
    };
    let Some(slice) = view.slice.as_ref() else {
        return Command::CancelDrag;
    };
    let Some(canvas) = map_drawable(area, view, graphics) else {
        return Command::CancelDrag;
    };
    if let Some(current) = view.zoom_bounds
        && !drag.zoom
        && let Some((rows, cols)) = drag.pan_delta(
            canvas,
            current.row_end.saturating_sub(current.row_start),
            current.col_end.saturating_sub(current.col_start),
        )
    {
        return Command::Pan { rows, cols };
    }
    let (rows, cols) = slice.values.dim();
    let flip_rows = slice
        .coordinates
        .as_ref()
        .is_some_and(|grid| grid.latitude_increases_with_source_row());
    if let Some(bounds) = drag.bounds_with_row_flip(canvas, rows, cols, flip_rows) {
        return Command::Zoom(Bounds {
            row_start: slice.source_bounds.row_start + bounds.row_start,
            row_end: slice.source_bounds.row_start + bounds.row_end,
            col_start: slice.source_bounds.col_start + bounds.col_start,
            col_end: slice.source_bounds.col_start + bounds.col_end,
        });
    }
    map_point_at(x, y, area, view, graphics)
        .map(|(row, col, _)| Command::SelectPoint { row, col })
        .unwrap_or(Command::CancelDrag)
}

/// Single source of truth for the sidebar Level section geometry used by the
/// mouse hit-test. `variable_rows` must match the widget's `.take(8)` count in
/// `sidebar.rs`, so the two agree on where the variable rows end.
fn level_geometry(
    sidebar: Rect,
    view: &ncview_rs::app::ViewModel,
    metadata: &DatasetMetadata,
    variable_query: &str,
) -> Option<ncview_rs::ui::level::LevelSection> {
    let plottable_all: Vec<_> = metadata
        .variables
        .iter()
        .filter(|variable| variable.numeric && plottable_variable(variable))
        .cloned()
        .collect();
    let filtered = ncview_rs::ui::sidebar::filter_variables(&plottable_all, variable_query);
    // The widget draws a single "no plottable fields" placeholder row when the
    // list is empty; match that so the section heading stays aligned.
    let variable_rows = if filtered.is_empty() {
        1
    } else {
        filtered.len().min(8)
    };
    let stepper_span = (usize::from(sidebar.width.saturating_sub(2)) / 2) as u16;
    let geometry = ncview_rs::ui::level::level_section(
        sidebar,
        variable_rows,
        view.level_labels.len(),
        stepper_span,
    )?;
    let (top, _) = ncview_rs::ui::level::level_window(
        view.level_labels.len(),
        geometry.list_rows,
        view.depth_cursor,
    );
    Some(ncview_rs::ui::level::LevelSection {
        window_top: top,
        ..geometry
    })
}

/// Map a click inside the sidebar panel to the command its row represents, or
/// `None` when the row carries no control. Sidebar rows: filename, colormap
/// name/scale, a View actions heading, two view-action rows, a Navigation
/// heading, two navigation rows, variables, the optional Level section,
/// dimensions, then the limit/filter controls.
fn translate_sidebar_position(
    x: u16,
    y: u16,
    sidebar: Rect,
    metadata: &DatasetMetadata,
    view: &ncview_rs::app::ViewModel,
    variable_query: &str,
) -> Option<Command> {
    if !sidebar.contains((x, y).into()) {
        return None;
    }
    let plottable_all: Vec<_> = metadata
        .variables
        .iter()
        .filter(|variable| variable.numeric && plottable_variable(variable))
        .cloned()
        .collect();
    let plottable = ncview_rs::ui::sidebar::filter_variables(&plottable_all, variable_query)
        .into_iter()
        .take(8)
        .collect::<Vec<_>>();
    let action_third = (sidebar.width / 3).max(1);
    let action_row_one = sidebar.y.saturating_add(6);
    let action_row_two = sidebar.y.saturating_add(7);
    if y == action_row_one {
        return Some(match (x.saturating_sub(sidebar.x)) / action_third {
            0 => Command::CyclePalette,
            1 => Command::TogglePaletteReverse,
            _ => Command::AutomaticLimits,
        });
    }
    if y == action_row_two {
        return Some(match (x.saturating_sub(sidebar.x)) / action_third {
            0 => Command::OpenLimits,
            1 => Command::OpenFilter,
            _ => Command::OpenAxisOverlay,
        });
    }
    let date_row = sidebar.y.saturating_add(9);
    if y == date_row {
        return Some(match (x.saturating_sub(sidebar.x)) / action_third {
            0 => Command::ResetZoom,
            1 => Command::MoveTime(-1),
            _ => Command::MoveTime(1),
        });
    }
    let speed_row = sidebar.y.saturating_add(10);
    if y == speed_row {
        return Some(match (x.saturating_sub(sidebar.x)) / action_third {
            0 => Command::DecreasePlaybackSpeed,
            1 => Command::IncreasePlaybackSpeed,
            _ => Command::ToggleColorScaleScope,
        });
    }
    let search_row = sidebar.y.saturating_add(12);
    if y == search_row {
        return Some(Command::OpenVariableSearch);
    }
    let variable_start = search_row.saturating_add(1);
    if y >= variable_start && usize::from(y - variable_start) < plottable.len() {
        return Some(Command::SelectVariableAt(usize::from(y - variable_start)));
    }
    let section = level_geometry(sidebar, view, metadata, variable_query);
    if let Some(section) = section.as_ref() {
        if y == section.stepper {
            return ncview_rs::ui::level::level_button(section, x).map(|button| match button {
                ncview_rs::ui::level::DepthButton::Prev => Command::MoveDepth(-1),
                ncview_rs::ui::level::DepthButton::Next => Command::MoveDepth(1),
            });
        }
        if let Some(index) = ncview_rs::ui::level::level_at(section, y) {
            return Some(Command::SetDepth(index));
        }
    }
    let dimensions = metadata.dimensions.iter().take(8).count();
    let level_rows = section.map_or(0, |section| {
        usize::from(section.list_top) + section.list_rows - usize::from(section.heading)
    });
    let dimensions_heading = variable_start
        .saturating_add(u16::try_from(plottable.len()).unwrap_or(u16::MAX))
        .saturating_add(1)
        .saturating_add(u16::try_from(level_rows).unwrap_or(u16::MAX));
    let dimensions_end = dimensions_heading
        .saturating_add(1)
        .saturating_add(u16::try_from(dimensions).unwrap_or(u16::MAX));
    let colormap_heading = sidebar.y.saturating_add(2);
    let palette_row = colormap_heading.saturating_add(1);
    let scale_row = colormap_heading.saturating_add(2);
    let limits_row = dimensions_end.saturating_add(1);
    let filter_row = dimensions_end.saturating_add(2);
    let scope_row = dimensions_end.saturating_add(4);
    let reverse_row = dimensions_end.saturating_add(5);
    let command = match y {
        value if value == palette_row => Command::CyclePalette,
        value if value == scale_row => Command::ToggleScale,
        value if value == limits_row => Command::OpenLimits,
        value if value == filter_row => Command::OpenFilter,
        value if value == scope_row => Command::ToggleColorScaleScope,
        value if value == reverse_row => {
            if x >= sidebar.x.saturating_add(18) {
                Command::ToggleLandBorders
            } else {
                Command::TogglePaletteReverse
            }
        }
        _ => return None,
    };
    Some(command)
}

/// Route a wheel scroll over the sidebar: the Level section steps depth, the
/// variable list cycles the selected variable. Outside the sidebar returns
/// `None` so the event stays a no-op.
fn translate_scroll(
    x: u16,
    y: u16,
    delta: isize,
    sidebar: Rect,
    metadata: &DatasetMetadata,
    view: &ncview_rs::app::ViewModel,
    variable_query: &str,
) -> Option<Command> {
    if !sidebar.contains((x, y).into()) {
        return None;
    }
    if let Some(section) = level_geometry(sidebar, view, metadata, variable_query) {
        let list_bottom = section
            .list_top
            .saturating_add(u16::try_from(section.list_rows.saturating_sub(1)).unwrap_or(u16::MAX));
        if y >= section.heading && y <= list_bottom {
            return Some(Command::MoveDepth(delta));
        }
    }
    Some(Command::SelectVariable(if delta < 0 { 0 } else { 1 }))
}

fn translate_mouse_position(
    command: Command,
    area: Rect,
    metadata: &DatasetMetadata,
    view: &ncview_rs::app::ViewModel,
    variable_query: &str,
    graphics: Option<&GraphicsRenderer>,
) -> Command {
    let (x, y, right, clicked) = match command {
        Command::MouseClick { x, y, right } => (x, y, right, true),
        Command::Pointer { x, y } => (x, y, false, false),
        command => return command,
    };
    if right {
        return Command::ToggleHelp;
    }
    if view.variable_search_active {
        return translate_variable_browser_click(x, y, clicked, area, metadata, variable_query);
    }
    if view.help_visible {
        return translate_help_click(x, y, clicked, area);
    }
    if let Some(overlay) = view.overlay {
        return translate_overlay_mouse(overlay, x, y, clicked, area, view);
    }
    if view.help_visible || view.overlay.is_some() {
        return Command::Pointer { x, y };
    }
    let areas = dashboard_layout::dashboard(area, view.depth_length > 1);
    if clicked && view.depth_length > 1 && areas.level.contains((x, y).into()) {
        return Command::SetDepth(depth_index_at(x, areas.level, view.depth_length));
    }
    if let Some((row, col, value)) = map_point_at(x, y, area, view, graphics) {
        return if clicked {
            Command::SelectPoint { row, col }
        } else if view
            .hover_point
            .as_ref()
            .is_some_and(|point| point.row == row && point.col == col)
        {
            // Mouse motion within the same source cell does not change the
            // scientific readout. Avoid replacing the hover state (and a
            // needless terminal diff) for every sub-cell cursor movement.
            Command::Pointer { x, y }
        } else {
            Command::HoverPoint {
                x,
                y,
                row,
                col,
                value,
            }
        };
    }
    if !clicked {
        return Command::ClearHover;
    }
    if areas.timeline.contains((x, y).into()) && view.time_length > 0 {
        if x <= areas.timeline.x.saturating_add(7) {
            return Command::TogglePlayback;
        }
        // The timeline speed controls are right-aligned in the panel title:
        // "[−] slower  [＋] faster". Give each control a generous hitbox.
        let speed_start = areas.timeline.right().saturating_sub(23);
        if x >= speed_start {
            return if x < speed_start.saturating_add(11) {
                Command::DecreasePlaybackSpeed
            } else {
                Command::IncreasePlaybackSpeed
            };
        }
        let start = areas.timeline.x.saturating_add(1);
        let end = areas
            .timeline
            .x
            .saturating_add(areas.timeline.width.saturating_sub(2));
        let position = x.clamp(start, end).saturating_sub(start) as usize;
        let width = usize::from(end.saturating_sub(start).max(1));
        let index = position.saturating_mul(view.time_length.saturating_sub(1)) / width;
        return Command::SetTime(index);
    }
    if !areas.sidebar.contains((x, y).into()) {
        return Command::Pointer { x, y };
    }
    translate_sidebar_position(x, y, areas.sidebar, metadata, view, variable_query)
        .unwrap_or(Command::Pointer { x, y })
}

fn translate_variable_browser_click(
    x: u16,
    y: u16,
    clicked: bool,
    area: Rect,
    metadata: &DatasetMetadata,
    variable_query: &str,
) -> Command {
    let popup = variable_browser_rect(area);
    if clicked {
        if close_button_hit(popup, x, y) || !popup.contains((x, y).into()) {
            return Command::Quit;
        }
        let inner_top = popup.y.saturating_add(3);
        if y >= inner_top {
            let index = usize::from(y - inner_top);
            let plottable = metadata
                .variables
                .iter()
                .filter(|variable| variable.numeric && plottable_variable(variable))
                .cloned()
                .collect::<Vec<_>>();
            let visible = ncview_rs::ui::sidebar::filter_variables(&plottable, variable_query);
            if index < visible.len() {
                return Command::SelectVariableAt(index);
            }
        }
    }
    Command::Pointer { x, y }
}

fn translate_help_click(x: u16, y: u16, clicked: bool, area: Rect) -> Command {
    let popup = help_rect(area);
    if clicked && (close_button_hit(popup, x, y) || !popup.contains((x, y).into())) {
        Command::ToggleHelp
    } else {
        Command::Pointer { x, y }
    }
}

fn translate_overlay_mouse(
    overlay: Overlay,
    x: u16,
    y: u16,
    clicked: bool,
    area: Rect,
    view: &ncview_rs::app::ViewModel,
) -> Command {
    let popup = overlay_rect(area, overlay);
    if clicked && (close_button_hit(popup, x, y) || !popup.contains((x, y).into())) {
        return Command::Quit;
    }
    match overlay {
        Overlay::CommandPalette => translate_command_palette_click(x, y, clicked, popup, view),
        Overlay::Limits | Overlay::Filter => translate_limit_overlay_click(x, y, clicked, popup),
        Overlay::Axis => translate_axis_overlay_click(x, y, clicked, popup),
        Overlay::Plot => translate_plot_overlay_click(x, y, clicked, popup),
        Overlay::TimeSeries => Command::Pointer { x, y },
    }
}

fn translate_command_palette_click(
    x: u16,
    y: u16,
    clicked: bool,
    popup: Rect,
    view: &ncview_rs::app::ViewModel,
) -> Command {
    if clicked {
        let inner_top = popup.y.saturating_add(3);
        if y >= inner_top {
            let index = usize::from(y - inner_top);
            let matches = ncview_rs::app::palette_matches(&view.palette_query);
            if index < matches.len() {
                return Command::ExecutePaletteChoice(index);
            }
        }
    }
    Command::Pointer { x, y }
}

fn translate_limit_overlay_click(x: u16, y: u16, clicked: bool, popup: Rect) -> Command {
    if clicked {
        if popup_row_hit(popup, x, y, 1) {
            return Command::FocusLimitField(LimitField::Min);
        }
        if popup_row_hit(popup, x, y, 2) {
            return Command::FocusLimitField(LimitField::Max);
        }
    }
    Command::Pointer { x, y }
}

fn translate_axis_overlay_click(x: u16, y: u16, clicked: bool, popup: Rect) -> Command {
    if clicked {
        if popup_row_hit(popup, x, y, 1) {
            return Command::FocusAxisField(AxisField::X);
        }
        if popup_row_hit(popup, x, y, 2) {
            return Command::FocusAxisField(AxisField::Y);
        }
    }
    Command::Pointer { x, y }
}

fn popup_row_hit(popup: Rect, x: u16, y: u16, row: u16) -> bool {
    x >= popup.x && x < popup.x.saturating_add(popup.width) && y == popup.y.saturating_add(row)
}

fn translate_plot_overlay_click(x: u16, y: u16, clicked: bool, popup: Rect) -> Command {
    if clicked {
        let content_x = popup.x.saturating_add(2);
        let type_y = popup.y.saturating_add(2);
        if y == type_y && x >= content_x && x < popup.right().saturating_sub(1) {
            let relative = x.saturating_sub(content_x);
            let fifth = (popup.width.saturating_sub(4) / 5).max(1);
            return if relative < fifth {
                Command::SetPlotKind(ncview_rs::app::PlotKind::TimeSeries)
            } else if relative < fifth.saturating_mul(2) {
                Command::SetPlotKind(ncview_rs::app::PlotKind::Scatter)
            } else if relative < fifth.saturating_mul(3) {
                Command::SetPlotKind(ncview_rs::app::PlotKind::Histogram)
            } else if relative < fifth.saturating_mul(4) {
                Command::SetPlotKind(ncview_rs::app::PlotKind::Cdf)
            } else {
                Command::SetPlotKind(ncview_rs::app::PlotKind::VerticalProfile)
            };
        }
        if x >= popup.x && x < popup.right() {
            if y == popup.y.saturating_add(4) {
                return Command::FocusPlotAxis(ncview_rs::app::PlotAxisField::X);
            }
            if y == popup.y.saturating_add(5) {
                return Command::FocusPlotAxis(ncview_rs::app::PlotAxisField::Y);
            }
        }
    }
    Command::Pointer { x, y }
}

fn close_button_hit(popup: Rect, x: u16, y: u16) -> bool {
    popup.width >= 4
        && y == popup.y
        && x >= popup.x.saturating_add(popup.width.saturating_sub(10))
        && x < popup.x.saturating_add(popup.width)
}

fn variable_browser_rect(area: Rect) -> Rect {
    let width = area.width.saturating_mul(4).saturating_div(5).max(1);
    let height = area.height.saturating_mul(4).saturating_div(5).max(1);
    Rect {
        x: area.x + area.width.saturating_sub(width.min(area.width)) / 2,
        y: area.y + area.height.saturating_sub(height.min(area.height)) / 2,
        width: width.min(area.width),
        height: height.min(area.height),
    }
}

fn help_rect(area: Rect) -> Rect {
    let width = area.width.saturating_mul(3) / 4;
    let height = area.height.saturating_mul(3) / 5;
    Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    }
}

fn overlay_rect(area: Rect, overlay: Overlay) -> Rect {
    let large = matches!(overlay, Overlay::CommandPalette | Overlay::Plot);
    let width = if large {
        area.width.saturating_mul(3) / 4
    } else {
        area.width.saturating_mul(3) / 5
    };
    let height = if large {
        area.height.saturating_mul(3) / 5
    } else {
        area.height.saturating_mul(2) / 5
    };
    Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    }
}

fn map_point_at(
    x: u16,
    y: u16,
    area: Rect,
    view: &ncview_rs::app::ViewModel,
    graphics: Option<&GraphicsRenderer>,
) -> Option<(usize, usize, Option<f64>)> {
    let slice = view.slice.as_ref()?;
    let drawable = map_drawable(area, view, graphics)?;
    let (rows, cols) = slice.values.dim();
    let drawable = if graphics.is_some_and(GraphicsRenderer::supports_graphics) {
        drawable
    } else {
        Rect::new(
            drawable.x,
            drawable.y,
            drawable.width.min(u16::try_from(cols).unwrap_or(u16::MAX)),
            drawable.height.min(u16::try_from(rows).unwrap_or(u16::MAX)),
        )
    };
    let flip_rows = slice
        .coordinates
        .as_ref()
        .is_some_and(|grid| grid.latitude_increases_with_source_row());
    let (row, col) =
        screen_to_source_with_row_flip(x, y, drawable, slice.source_bounds, flip_rows)?;
    let value = slice.value_at_source(row, col);
    Some((row, col, value))
}

fn map_drawable(
    area: Rect,
    view: &ncview_rs::app::ViewModel,
    graphics: Option<&GraphicsRenderer>,
) -> Option<Rect> {
    let _ = view.slice.as_ref()?;
    let panel = dashboard_layout::dashboard(area, view.depth_length > 1).canvas;
    let inner = Rect::new(
        panel.x.saturating_add(1),
        panel.y.saturating_add(1),
        panel.width.saturating_sub(2),
        panel.height.saturating_sub(2),
    );
    if graphics.is_some_and(GraphicsRenderer::supports_graphics) {
        return Some(graphics.map_or(inner, |renderer| renderer.drawable_area(inner)));
    }
    {
        let slice = view.slice.as_ref()?;
        let (rows, cols) = slice.values.dim();
        Some(Rect::new(
            inner.x,
            inner.y,
            inner.width.min(u16::try_from(cols).unwrap_or(u16::MAX)),
            inner.height.min(u16::try_from(rows).unwrap_or(u16::MAX)),
        ))
    }
}

fn select_initial_variable(state: &mut AppState, metadata: &DatasetMetadata) {
    let selected = state
        .variables
        .iter()
        .filter(|variable| variable.numeric && plottable_variable(variable))
        .max_by_key(|variable| {
            let area_penalty = variable.name.to_ascii_lowercase().contains("area");
            (variable.dimensions.len(), !area_penalty)
        })
        .map(|variable| variable.name.clone());
    state.view.selected_variable = selected;
    if let Some(variable) = state.view.selected_variable.clone()
        && let Some(metadata_variable) =
            metadata.variables.iter().find(|item| item.name == variable)
    {
        state.view.axis_options = if data::is_mesh_variable(metadata_variable) {
            vec!["latitude".into(), "longitude".into()]
        } else {
            metadata_variable.dimensions.clone()
        };
        if data::is_mesh_variable(metadata_variable) {
            state.view.x_axis = Some("longitude".into());
            state.view.y_axis = Some("latitude".into());
        }
        let (time_length, depth_length) = leading_lengths(metadata, metadata_variable);
        state.view.time_length = time_length;
        state.view.depth_length = depth_length;
        state.view.status = format!(
            "opened {} variable(s); selected {variable}",
            state.variables.len()
        );
    }
}

fn plottable_variable(variable: &Variable) -> bool {
    variable.dimensions.len() >= 2 || data::is_mesh_variable(variable)
}

fn earliest_source_index(sources: &[Arc<dyn data::DataSource>]) -> usize {
    sources
        .iter()
        .enumerate()
        .min_by(|(left_index, left), (right_index, right)| {
            match (
                source_earliest_time(left.as_ref()),
                source_earliest_time(right.as_ref()),
            ) {
                (Some(left_time), Some(right_time)) => left_time
                    .cmp(&right_time)
                    .then_with(|| left_index.cmp(right_index)),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => left_index.cmp(right_index),
            }
        })
        .map_or(0, |(index, _)| index)
}

fn source_earliest_time(source: &dyn data::DataSource) -> Option<DateTime<Utc>> {
    source
        .metadata()
        .variables
        .iter()
        .filter(|variable| variable.numeric && plottable_variable(variable))
        .flat_map(|variable| {
            let time_length = leading_lengths(source.metadata(), variable).0.max(1);
            (0..time_length).filter_map(|index| {
                source
                    .time_label_for_variable(&variable.name, index)
                    .and_then(|label| parse_time_label(&label))
            })
        })
        .min()
}

fn parse_time_label(label: &str) -> Option<DateTime<Utc>> {
    let normalized = label
        .strip_suffix('z')
        .map_or_else(|| label.to_owned(), |prefix| format!("{prefix}Z"));
    DateTime::parse_from_rfc3339(&normalized)
        .ok()
        .map(|time| time.with_timezone(&Utc))
}

fn compare_time_labels(left: &str, right: &str) -> std::cmp::Ordering {
    match (parse_time_label(left), parse_time_label(right)) {
        (Some(left_time), Some(right_time)) => left_time.cmp(&right_time),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

fn configure_timeline(
    state: &mut AppState,
    sources: &[Arc<dyn data::DataSource>],
    manifest: &VirtualDatasetManifest,
    active_file: usize,
) {
    state.view.timeline.clear();
    let Some(variable_name) = state.view.selected_variable.as_deref() else {
        state.view.time_length = 1;
        state.view.time_index = 0;
        state.view.time_label = None;
        return;
    };
    for frame in manifest
        .frames_for_variable(variable_name)
        .into_iter()
        .flatten()
    {
        let source_index = frame.source_index;
        let local_index = frame.local_index;
        if manifest
            .variable(variable_name)
            .is_some_and(|variable| !variable.compatible && source_index != active_file)
        {
            // A source with incompatible schema or coordinates is isolated to
            // its explicitly selected file. The manifest retains the reason
            // so the UI can expose it without combining unsafe frames.
            continue;
        }
        let Some(source) = sources.get(source_index) else {
            continue;
        };
        if !source
            .metadata()
            .variables
            .iter()
            .any(|variable| variable.name == variable_name)
        {
            continue;
        }
        let global_index = state.view.timeline.len();
        let label = frame
            .label
            .clone()
            .or_else(|| source.time_label_for_variable(variable_name, local_index))
            .unwrap_or_else(|| format!("t={global_index}"));
        state.view.timeline.push(TimelinePoint {
            source_index,
            local_index,
            label,
        });
    }
    state.view.timeline.sort_by(|left, right| {
        compare_time_labels(&left.label, &right.label)
            .then_with(|| left.source_index.cmp(&right.source_index))
            .then_with(|| left.local_index.cmp(&right.local_index))
    });
    if state.view.timeline.is_empty()
        && let Some(source) = sources.get(active_file)
        && source
            .metadata()
            .variables
            .iter()
            .any(|variable| variable.name == variable_name)
    {
        state.view.timeline.push(TimelinePoint {
            source_index: active_file,
            local_index: 0,
            label: "coordinate index".into(),
        });
    }
    state.view.time_length = state.view.timeline.len().max(1);
    state.view.time_index = state
        .view
        .time_index
        .min(state.view.time_length.saturating_sub(1));
    state.view.time_label = state
        .view
        .timeline
        .get(state.view.time_index)
        .map(|point| point.label.clone());
}

fn load_selected(
    state: &mut AppState,
    sources: &[Arc<dyn data::DataSource>],
    active_file: usize,
    slice_tx: &std::sync::mpsc::Sender<RemoteSliceMessage>,
    slice_cancelled: &mut Arc<AtomicBool>,
) {
    // Every new navigation request supersedes the previous read. Local files
    // can be just as expensive as remote objects (metadata-backed NetCDF
    // hyperslabs and GRIB decoding both touch a lot of bytes), so all slice
    // work stays behind the worker boundary. The generation check below keeps
    // an older worker from replacing the last accepted image.
    slice_cancelled.store(true, Ordering::Release);
    let Some(variable_name) = state.view.selected_variable.clone() else {
        return;
    };
    let timeline_point = state
        .view
        .timeline
        .get(state.view.time_index)
        .cloned()
        .unwrap_or(TimelinePoint {
            source_index: active_file,
            local_index: 0,
            label: "coordinate index".into(),
        });
    let Some(source) = sources.get(timeline_point.source_index) else {
        return;
    };
    let metadata = source.metadata();
    let Some(variable) = metadata
        .variables
        .iter()
        .find(|item| item.name == variable_name)
    else {
        return;
    };
    let Some((full_bounds, _time_length, depth_length)) = plane_bounds(
        metadata,
        variable,
        state.view.x_axis.as_deref(),
        state.view.y_axis.as_deref(),
    ) else {
        state.view.status = format!("{variable_name}: needs at least two dimensions");
        return;
    };
    state.view.time_length = state.view.timeline.len().max(1);
    state.view.depth_length = depth_length;
    state.view.depth_index = state.view.depth_index.min(depth_length.saturating_sub(1));
    state.view.time_label = Some(timeline_point.label.clone());
    state.view.level_label = source.vertical_label(&variable_name, state.view.depth_index);
    state.view.level_labels = source.vertical_labels(&variable_name);
    state.view.depth_cursor = state.view.depth_index;
    state.view.full_bounds = Some(full_bounds);
    let bounds = state.view.zoom_bounds.unwrap_or(full_bounds);
    let fixed_axes = fixed_axes_for_plane(
        metadata,
        variable,
        state.view.x_axis.as_deref(),
        state.view.y_axis.as_deref(),
        timeline_point.local_index,
        state.view.depth_index,
    );
    let request = SliceRequest {
        variable: variable_name.clone(),
        time: timeline_point.local_index,
        depth: state.view.depth_index,
        bounds,
    };
    let request_cancelled = Arc::new(AtomicBool::new(false));
    *slice_cancelled = Arc::clone(&request_cancelled);
    let generation = state.next_generation();
    state.view.loading = ncview_rs::app::LoadingState::Loading;
    state.view.status = format!("{variable_name}: loading slice…");
    let source = Arc::clone(source);
    let tx = slice_tx.clone();
    let variable = variable_name.clone();
    let row_axis = state.view.y_axis.clone();
    let col_axis = state.view.x_axis.clone();
    let fixed_axes_for_read = fixed_axes;
    let full_request = SliceRequest {
        variable: variable_name.clone(),
        time: timeline_point.local_index,
        depth: state.view.depth_index,
        bounds: full_bounds,
    };
    let full_view = bounds == full_bounds;
    let global_view = state.view.color_scale_scope == ColorScaleScope::GlobalView;
    let scale_mode = state.view.scale_mode;
    let source_index = timeline_point.source_index;
    std::thread::spawn(move || {
        let result = catch_unwind(AssertUnwindSafe(|| {
            if request_cancelled.load(Ordering::Acquire) {
                return Err("slice superseded before read".to_owned());
            }
            let slice = source
                .read_slice_on_axes_cancellable(
                    &request,
                    row_axis.as_deref(),
                    col_axis.as_deref(),
                    &fixed_axes_for_read,
                    Arc::clone(&request_cancelled),
                )
                .map_err(|error| error.to_string())?;
            if request_cancelled.load(Ordering::Acquire) {
                return Err("slice superseded".to_owned());
            }
            let full_limits = if !full_view && global_view {
                if request_cancelled.load(Ordering::Acquire) {
                    return Err("slice superseded before limit read".to_owned());
                }
                source
                    .read_slice_on_axes_cancellable(
                        &full_request,
                        row_axis.as_deref(),
                        col_axis.as_deref(),
                        &fixed_axes_for_read,
                        Arc::clone(&request_cancelled),
                    )
                    .ok()
                    .and_then(|full_slice| slice_limits(&full_slice, scale_mode))
            } else {
                None
            };
            Ok::<_, String>((slice, full_limits))
        }))
        .map_err(|_| format!("{variable}: slice worker panicked"))
        .and_then(|result| result);
        let _ = tx.send(RemoteSliceMessage {
            generation,
            source_index,
            variable,
            full_view,
            result,
        });
    });
}

fn apply_remote_slice(
    state: &mut AppState,
    sources: &[Arc<dyn data::DataSource>],
    message: RemoteSliceMessage,
) {
    if message.generation != state.view.generation {
        return;
    }
    let Some(source) = sources.get(message.source_index) else {
        return;
    };
    match message.result {
        Ok((slice, full_limits)) => {
            if slice.memory_bytes() > state.view.decoded_limit {
                state.view.loading = ncview_rs::app::LoadingState::Error;
                state.view.status = format!(
                    "{}: decoded slice exceeds working-set limit ({} bytes)",
                    message.variable, state.view.decoded_limit
                );
                return;
            }
            let current_limits = slice_limits(&slice, state.view.scale_mode);
            if message.full_view {
                state.view.global_limits = current_limits;
            } else if state.view.color_scale_scope == ColorScaleScope::GlobalView
                && let Some(full_limits) = full_limits
            {
                state.view.global_limits = Some(full_limits);
            }
            if !state.view.limits_manual && state.view.limits.is_none() {
                state.view.limits = match state.view.color_scale_scope {
                    ColorScaleScope::CurrentView => current_limits,
                    ColorScaleScope::GlobalView => {
                        state.view.global_limits.or(full_limits).or(current_limits)
                    }
                };
            }
            if !state.accept_slice(message.generation, slice) {
                return;
            }
            if let Some(point) = state.view.hover_point.as_mut()
                && let Some(value) = state
                    .view
                    .slice
                    .as_ref()
                    .and_then(|slice| slice.value_at_source(point.row, point.col))
            {
                point.value = Some(value);
            }
            let time_text = state
                .view
                .time_label
                .as_deref()
                .unwrap_or("coordinate index");
            let level_text = state
                .view
                .level_label
                .as_deref()
                .map_or_else(String::new, |label| format!("  level={label}"));
            state.view.status =
                format!("{}  time={time_text}{level_text}  ready", message.variable);
        }
        Err(error) => {
            // Keep the last valid slice visible while the remote request fails.
            state.view.loading = ncview_rs::app::LoadingState::Error;
            state.view.status = format!("{}: {error}", message.variable);
            let _ = source;
        }
    }
}

fn apply_remote_plot(state: &mut AppState, message: RemotePlotMessage) {
    if message.generation != state.view.plot_generation {
        return;
    }
    match message.result {
        Ok(result) => {
            let _ = state.accept_plot(
                message.generation,
                result.plot_series,
                result.time_series,
                result.time_series_labels,
                result.status,
            );
        }
        Err(error) => {
            // Keep the previous valid plot visible while a replacement is
            // cancelled or fails, matching the map's last-valid-frame policy.
            state.view.status = format!("plot: {error}");
        }
    }
}

fn slice_limits(
    slice: &ncview_rs::data::slice::Slice2D,
    scale: ncview_rs::app::ScaleMode,
) -> Option<(f64, f64)> {
    if scale == ncview_rs::app::ScaleMode::Log {
        ncview_rs::app::positive_slice_limits(slice)
    } else {
        slice.statistics.map(|stats| (stats.min, stats.max))
    }
}

fn load_time_series(
    state: &mut AppState,
    sources: &[Arc<dyn data::DataSource>],
    active_file: usize,
    plot_tx: &std::sync::mpsc::Sender<RemotePlotMessage>,
    plot_cancelled: &mut Arc<AtomicBool>,
) {
    plot_cancelled.store(true, Ordering::Release);
    let request_cancelled = Arc::new(AtomicBool::new(false));
    *plot_cancelled = Arc::clone(&request_cancelled);
    let generation = state.next_plot_generation();
    state.view.status = "loading plot data…".into();
    let mut working_state = plot_worker_state(state);
    let sources = sources.to_vec();
    let tx = plot_tx.clone();
    std::thread::spawn(move || {
        if request_cancelled.load(Ordering::Acquire) {
            return;
        }
        let result = catch_unwind(AssertUnwindSafe(|| {
            compute_time_series(
                &mut working_state,
                &sources,
                active_file,
                Some(&request_cancelled),
            );
            RemotePlotResult {
                plot_series: working_state.view.plot_series,
                time_series: working_state.view.time_series,
                time_series_labels: working_state.view.time_series_labels,
                status: working_state.view.status,
            }
        }))
        .map_err(|_| "plot worker panicked".to_owned())
        .and_then(|result| {
            if request_cancelled.load(Ordering::Acquire) {
                Err("plot request superseded".to_owned())
            } else {
                Ok(result)
            }
        });
        let _ = tx.send(RemotePlotMessage { generation, result });
    });
}

fn plot_worker_state(state: &AppState) -> AppState {
    let mut worker = AppState::default();
    worker.view.selected_variable = state.view.selected_variable.clone();
    worker.view.selected_point = state.view.selected_point;
    worker.view.selected_points = state.view.selected_points.clone();
    worker.view.plot_draft = state.view.plot_draft;
    worker.view.timeline = state.view.timeline.clone();
    worker.view.depth_index = state.view.depth_index;
    worker.view.time_index = state.view.time_index;
    worker.view.level_label = state.view.level_label.clone();
    worker.view.zoom_bounds = state.view.zoom_bounds;
    worker.view.x_axis = state.view.x_axis.clone();
    worker.view.y_axis = state.view.y_axis.clone();
    worker
}

fn compute_time_series(
    state: &mut AppState,
    sources: &[Arc<dyn data::DataSource>],
    active_file: usize,
    cancelled: Option<&AtomicBool>,
) {
    state.view.time_series.clear();
    state.view.time_series_labels.clear();
    state.view.plot_series.clear();
    let selected_points = if state.view.selected_points.is_empty() {
        state.view.selected_point.into_iter().collect::<Vec<_>>()
    } else {
        state.view.selected_points.clone()
    };
    if selected_points.is_empty() {
        load_domain_summary(state, sources, cancelled);
        return;
    }
    if matches!(
        state.view.plot_draft.x_axis,
        ncview_rs::app::PlotXAxis::Longitude
            | ncview_rs::app::PlotXAxis::Latitude
            | ncview_rs::app::PlotXAxis::Dimension(_)
    ) {
        load_cross_section(state, sources, &selected_points, cancelled);
        return;
    }
    let Some(variable_name) = state.view.selected_variable.clone() else {
        return;
    };
    let Some(active_source) = sources.get(active_file) else {
        return;
    };
    let timeline = state.view.timeline.clone();
    let depth_index = state.view.depth_index;
    let mut used_labels = Vec::new();
    let mut total_finite = 0;

    for (point_number, (row, col)) in selected_points.iter().copied().enumerate() {
        let mut data = Vec::with_capacity(timeline.len());
        let mut labels = Vec::with_capacity(timeline.len());
        let mut finite_samples = 0;
        for (time_index, point) in timeline.iter().enumerate() {
            if cancelled.is_some_and(|token| token.load(Ordering::Acquire)) {
                return;
            }
            let Some(source) = sources.get(point.source_index) else {
                data.push((time_index as f64, f64::NAN));
                labels.push(point.label.clone());
                continue;
            };
            let Some(variable) = source
                .metadata()
                .variables
                .iter()
                .find(|variable| variable.name == variable_name)
            else {
                data.push((time_index as f64, f64::NAN));
                labels.push(point.label.clone());
                continue;
            };
            let Some((source_bounds, _, depth_length)) =
                spatial_bounds(source.metadata(), variable)
            else {
                data.push((time_index as f64, f64::NAN));
                labels.push(point.label.clone());
                continue;
            };
            if row < source_bounds.row_start
                || row >= source_bounds.row_end
                || col < source_bounds.col_start
                || col >= source_bounds.col_end
            {
                data.push((time_index as f64, f64::NAN));
                labels.push(point.label.clone());
                continue;
            }
            let request = SliceRequest {
                variable: variable_name.clone(),
                time: point.local_index,
                depth: depth_index.min(depth_length.saturating_sub(1)),
                bounds: Bounds::new(row, row + 1, col, col + 1)
                    .expect("point bounds are non-empty"),
            };
            let value = source
                .read_slice_on_axes(&request, None, None, &[])
                .ok()
                .and_then(|slice| slice.value_at_source(row, col))
                .unwrap_or(f64::NAN);
            finite_samples += usize::from(value.is_finite());
            data.push((time_index as f64, value));
            labels.push(point.label.clone());
        }
        total_finite += finite_samples;
        let base_label = point_label(active_source.as_ref(), &variable_name, row, col);
        let label = if used_labels.contains(&base_label) {
            format!("{base_label} #{}", point_number + 1)
        } else {
            base_label
        };
        used_labels.push(label.clone());
        state.view.plot_series.push(PlotSeries {
            point: (row, col),
            label,
            data,
            labels,
        });
    }

    if let Some(first) = state.view.plot_series.first() {
        state.view.time_series = first.data.clone();
        state.view.time_series_labels = first.labels.clone();
    }
    let sample_count = state
        .view
        .plot_series
        .first()
        .map_or(0, |series| series.data.len());
    state.view.status = format!(
        "{} point(s): {total_finite}/{} finite timeline samples",
        selected_points.len(),
        sample_count.saturating_mul(selected_points.len())
    );
}

fn load_cross_section(
    state: &mut AppState,
    sources: &[Arc<dyn data::DataSource>],
    selected_points: &[(usize, usize)],
    cancelled: Option<&AtomicBool>,
) {
    let Some(variable_name) = state.view.selected_variable.clone() else {
        return;
    };
    let x_axis = state.view.plot_draft.x_axis;
    let Some(time_point) = state.view.timeline.get(state.view.time_index) else {
        return;
    };
    let Some(source) = sources.get(time_point.source_index) else {
        return;
    };
    let Some(variable) = source
        .metadata()
        .variables
        .iter()
        .find(|variable| variable.name == variable_name)
    else {
        return;
    };
    let Some((source_bounds, _, depth_length)) = spatial_bounds(source.metadata(), variable) else {
        return;
    };
    let depth = state.view.depth_index.min(depth_length.saturating_sub(1));
    let Some(x_dimension) = plot_dimension_name(source.metadata(), variable, x_axis) else {
        state.view.status = "the selected plot dimension is unavailable for this field".into();
        return;
    };
    let Some(x_length) = dimension_length(source.metadata(), &x_dimension) else {
        return;
    };
    let dimension_coordinates = source.dimension_values(&variable_name, &x_dimension);
    let fixed_dimension = variable
        .dimensions
        .iter()
        .find(|name| {
            !name.eq_ignore_ascii_case(&x_dimension)
                && (state.view.x_axis.as_deref() == Some(name.as_str())
                    || state.view.y_axis.as_deref() == Some(name.as_str()))
        })
        .or_else(|| {
            variable
                .dimensions
                .iter()
                .find(|name| !name.eq_ignore_ascii_case(&x_dimension))
        })
        .cloned();
    let Some(fixed_dimension) = fixed_dimension else {
        state.view.status = "a cross-section needs at least two dimensions".into();
        return;
    };
    let mut series = Vec::new();
    let mut used_labels = Vec::new();
    let level_label = state
        .view
        .level_label
        .clone()
        .unwrap_or_else(|| format!("depth={depth}"));

    for (point_number, &(row, col)) in selected_points.iter().enumerate() {
        if cancelled.is_some_and(|token| token.load(Ordering::Acquire)) {
            return;
        }
        if let Some(mut point_series) = cross_section_series(
            source.as_ref(),
            &variable_name,
            variable,
            source_bounds,
            &x_dimension,
            x_length,
            dimension_coordinates.as_deref(),
            &fixed_dimension,
            time_point,
            depth,
            &level_label,
            row,
            col,
            cancelled,
        ) {
            let base = point_series.label.clone();
            point_series.label = if used_labels.contains(&base) {
                format!("{base} #{}", point_number + 1)
            } else {
                base
            };
            used_labels.push(point_series.label.clone());
            series.push(point_series);
        }
        if cancelled.is_some_and(|token| token.load(Ordering::Acquire)) {
            return;
        }
    }
    if let Some(first) = series.first() {
        state.view.time_series = first.data.clone();
        state.view.time_series_labels.clear();
    }
    state.view.plot_series = series;
    state.view.status = format!(
        "{x_dimension} cross-section at time {} and {level_label}",
        time_point.label
    );
}

#[allow(clippy::too_many_arguments)]
fn cross_section_series(
    source: &dyn data::DataSource,
    variable_name: &str,
    variable: &Variable,
    source_bounds: Bounds,
    x_dimension: &str,
    x_length: usize,
    dimension_coordinates: Option<&[f64]>,
    fixed_dimension: &str,
    time_point: &TimelinePoint,
    depth: usize,
    level_label: &str,
    row: usize,
    col: usize,
    cancelled: Option<&AtomicBool>,
) -> Option<PlotSeries> {
    let x_is_grib = source.metadata().format == DatasetFormat::Grib2;
    let latitude_dimension = plot_dimension_name(
        source.metadata(),
        variable,
        ncview_rs::app::PlotXAxis::Latitude,
    )
    .unwrap_or_else(|| "latitude".into());
    let longitude_dimension = plot_dimension_name(
        source.metadata(),
        variable,
        ncview_rs::app::PlotXAxis::Longitude,
    )
    .unwrap_or_else(|| "longitude".into());
    let x_on_row = !x_is_grib || x_dimension.eq_ignore_ascii_case(&latitude_dimension);
    let fixed_index = dimension_index_for_plot(
        source.metadata(),
        fixed_dimension,
        row,
        col,
        time_point.local_index,
        depth,
    );
    let bounds = if x_is_grib {
        if x_on_row {
            Bounds::new(source_bounds.row_start, source_bounds.row_end, col, col + 1)
        } else {
            Bounds::new(row, row + 1, source_bounds.col_start, source_bounds.col_end)
        }
    } else if x_on_row {
        Bounds::new(0, x_length, fixed_index, fixed_index + 1)
    } else {
        Bounds::new(fixed_index, fixed_index + 1, 0, x_length)
    }
    .ok()?;
    if row < source_bounds.row_start
        || row >= source_bounds.row_end
        || col < source_bounds.col_start
        || col >= source_bounds.col_end
    {
        return None;
    }
    let request = SliceRequest {
        variable: variable_name.to_owned(),
        time: time_point.local_index,
        depth,
        bounds,
    };
    let fixed_axes = variable
        .dimensions
        .iter()
        .filter(|name| {
            !name.eq_ignore_ascii_case(x_dimension) && !name.eq_ignore_ascii_case(fixed_dimension)
        })
        .map(|name| {
            (
                name.clone(),
                dimension_index_for_plot(
                    source.metadata(),
                    name,
                    row,
                    col,
                    time_point.local_index,
                    depth,
                ),
            )
        })
        .collect::<Vec<_>>();
    let row_axis = if x_is_grib {
        Some(latitude_dimension.as_str())
    } else if x_on_row {
        Some(x_dimension)
    } else {
        Some(fixed_dimension)
    };
    let col_axis = if x_is_grib {
        Some(longitude_dimension.as_str())
    } else if x_on_row {
        Some(fixed_dimension)
    } else {
        Some(x_dimension)
    };
    let slice = source
        .read_slice_on_axes(&request, row_axis, col_axis, &fixed_axes)
        .ok()?;
    let count = if x_is_grib {
        if x_on_row {
            source_bounds.row_end - source_bounds.row_start
        } else {
            source_bounds.col_end - source_bounds.col_start
        }
    } else {
        x_length
    };
    let mut data = Vec::with_capacity(count);
    for offset in 0..count {
        if cancelled.is_some_and(|token| token.load(Ordering::Acquire)) {
            return None;
        }
        let (sample_row, sample_col) = if x_is_grib {
            if x_on_row {
                (source_bounds.row_start + offset, col)
            } else {
                (row, source_bounds.col_start + offset)
            }
        } else if x_on_row {
            (offset, fixed_index)
        } else {
            (fixed_index, offset)
        };
        let coordinate = source.point_coordinates(
            variable_name,
            if x_dimension.eq_ignore_ascii_case(&latitude_dimension) {
                if x_on_row { offset } else { row }
            } else {
                row
            },
            if x_dimension.eq_ignore_ascii_case(&longitude_dimension) {
                if x_on_row { offset } else { col }
            } else {
                col
            },
        );
        let x = if x_dimension.eq_ignore_ascii_case(&longitude_dimension) {
            coordinate
                .longitude
                .filter(|value| value.is_finite())
                .unwrap_or(offset as f64)
        } else if x_dimension.eq_ignore_ascii_case(&latitude_dimension) {
            coordinate
                .latitude
                .filter(|value| value.is_finite())
                .unwrap_or(offset as f64)
        } else {
            dimension_coordinates
                .and_then(|values| values.get(offset).copied())
                .filter(|value| value.is_finite())
                .unwrap_or(offset as f64)
        };
        let y = slice
            .value_at_source(sample_row, sample_col)
            .unwrap_or(f64::NAN);
        data.push((x, y));
    }
    data.sort_by(|left, right| left.0.total_cmp(&right.0));
    Some(PlotSeries {
        point: (row, col),
        label: format!(
            "{} @ {level_label}",
            point_label(source, variable_name, row, col)
        ),
        data,
        labels: Vec::new(),
    })
}

fn plot_dimension_name(
    metadata: &DatasetMetadata,
    variable: &Variable,
    x_axis: ncview_rs::app::PlotXAxis,
) -> Option<String> {
    match x_axis {
        ncview_rs::app::PlotXAxis::Longitude => variable
            .dimensions
            .iter()
            .find(|name| {
                metadata
                    .dimensions
                    .iter()
                    .find(|dimension| dimension.name.eq_ignore_ascii_case(name))
                    .is_some_and(|dimension| dimension.role == AxisRole::Longitude)
                    || name.to_ascii_lowercase().contains("lon")
            })
            .cloned(),
        ncview_rs::app::PlotXAxis::Latitude => variable
            .dimensions
            .iter()
            .find(|name| {
                metadata
                    .dimensions
                    .iter()
                    .find(|dimension| dimension.name.eq_ignore_ascii_case(name))
                    .is_some_and(|dimension| dimension.role == AxisRole::Latitude)
                    || name.to_ascii_lowercase().contains("lat")
            })
            .cloned(),
        ncview_rs::app::PlotXAxis::Dimension(index) => variable.dimensions.get(index).cloned(),
        _ => None,
    }
}

fn dimension_index_for_plot(
    metadata: &DatasetMetadata,
    dimension_name: &str,
    row: usize,
    col: usize,
    time: usize,
    depth: usize,
) -> usize {
    let dimension = metadata
        .dimensions
        .iter()
        .find(|dimension| dimension.name.eq_ignore_ascii_case(dimension_name));
    let index = match dimension.map(|dimension| dimension.role) {
        Some(AxisRole::Latitude) => row,
        Some(AxisRole::Longitude) => col,
        Some(AxisRole::Time) => time,
        Some(AxisRole::Depth) => depth,
        _ => 0,
    };
    dimension
        .map(|dimension| index.min(dimension.length.saturating_sub(1)))
        .unwrap_or(index)
}

fn point_label(
    source: &dyn data::DataSource,
    variable_name: &str,
    row: usize,
    col: usize,
) -> String {
    let coordinates = source.point_coordinates(variable_name, row, col);
    match (coordinates.latitude, coordinates.longitude) {
        (Some(latitude), Some(longitude)) => format!("lat={latitude:.2}, lon={longitude:.2}"),
        _ => format!("row={row}, col={col}"),
    }
}

fn load_domain_summary(
    state: &mut AppState,
    sources: &[Arc<dyn data::DataSource>],
    cancelled: Option<&AtomicBool>,
) {
    let Some(variable_name) = state.view.selected_variable.clone() else {
        return;
    };
    let timeline = state.view.timeline.clone();
    let depth_index = state.view.depth_index;
    let view_bounds = state.view.zoom_bounds;
    let mut mean = Vec::with_capacity(timeline.len());
    let mut minimum = Vec::with_capacity(timeline.len());
    let mut maximum = Vec::with_capacity(timeline.len());
    let mut labels = Vec::with_capacity(timeline.len());
    let mut finite_samples = 0;

    for point in &timeline {
        if cancelled.is_some_and(|token| token.load(Ordering::Acquire)) {
            return;
        }
        let Some(source) = sources.get(point.source_index) else {
            continue;
        };
        let Some(variable) = source
            .metadata()
            .variables
            .iter()
            .find(|variable| variable.name == variable_name)
        else {
            continue;
        };
        let Some((source_bounds, _, depth_length)) = spatial_bounds(source.metadata(), variable)
        else {
            continue;
        };
        let Some(bounds) = clamp_domain_bounds(source_bounds, view_bounds) else {
            continue;
        };
        let request = SliceRequest {
            variable: variable_name.clone(),
            time: point.local_index,
            depth: depth_index.min(depth_length.saturating_sub(1)),
            bounds,
        };
        let statistics = source
            .read_slice_on_axes(&request, None, None, &[])
            .ok()
            .and_then(|slice| slice.statistics);
        if let Some(statistics) = statistics {
            finite_samples += statistics.finite_count;
            mean.push((mean.len() as f64, statistics.mean));
            minimum.push((minimum.len() as f64, statistics.min));
            maximum.push((maximum.len() as f64, statistics.max));
        } else {
            let index = mean.len() as f64;
            mean.push((index, f64::NAN));
            minimum.push((index, f64::NAN));
            maximum.push((index, f64::NAN));
        }
        labels.push(point.label.clone());
    }

    state.view.plot_series = vec![
        PlotSeries {
            point: (0, 0),
            label: "mean".into(),
            data: mean.clone(),
            labels: labels.clone(),
        },
        PlotSeries {
            point: (0, 0),
            label: "minimum".into(),
            data: minimum,
            labels: labels.clone(),
        },
        PlotSeries {
            point: (0, 0),
            label: "maximum".into(),
            data: maximum,
            labels: labels.clone(),
        },
    ];
    state.view.time_series = mean;
    state.view.time_series_labels = labels;
    let domain_label = if view_bounds.is_some() {
        "view window"
    } else {
        "full field"
    };
    state.view.status = format!(
        "domain summary ({domain_label}): {finite_samples} finite values across {} timeline samples",
        state.view.time_series.len()
    );
}

fn clamp_domain_bounds(source_bounds: Bounds, requested: Option<Bounds>) -> Option<Bounds> {
    let requested = requested.unwrap_or(source_bounds);
    let row_start = requested
        .row_start
        .max(source_bounds.row_start)
        .min(source_bounds.row_end.saturating_sub(1));
    let row_end = requested.row_end.min(source_bounds.row_end);
    let col_start = requested
        .col_start
        .max(source_bounds.col_start)
        .min(source_bounds.col_end.saturating_sub(1));
    let col_end = requested.col_end.min(source_bounds.col_end);
    Bounds::new(row_start, row_end, col_start, col_end).ok()
}

fn leading_lengths(metadata: &DatasetMetadata, variable: &Variable) -> (usize, usize) {
    let row_index = variable
        .dimensions
        .iter()
        .position(|name| {
            metadata
                .dimensions
                .iter()
                .find(|dimension| dimension.name == *name)
                .is_some_and(|dimension| dimension.role == AxisRole::Latitude)
        })
        .unwrap_or(variable.dimensions.len().saturating_sub(2));
    let col_index = variable
        .dimensions
        .iter()
        .position(|name| {
            metadata
                .dimensions
                .iter()
                .find(|dimension| dimension.name == *name)
                .is_some_and(|dimension| dimension.role == AxisRole::Longitude)
        })
        .unwrap_or(variable.dimensions.len().saturating_sub(1));
    variable
        .dimensions
        .iter()
        .enumerate()
        .filter(|(axis, _)| *axis != row_index && *axis != col_index)
        .fold((1, 1), |(time, depth), (axis, name)| {
            let length = metadata
                .dimensions
                .iter()
                .find(|dimension| dimension.name == *name)
                .map_or(1, |dimension| dimension.length);
            let role = match metadata
                .dimensions
                .iter()
                .find(|dimension| dimension.name == *name)
                .map(|dimension| dimension.role)
            {
                Some(AxisRole::Time) => AxisRole::Time,
                Some(AxisRole::Depth) => AxisRole::Depth,
                _ if axis == row_index => AxisRole::Other,
                _ => match axis {
                    0 => AxisRole::Time,
                    1 => AxisRole::Depth,
                    _ => AxisRole::Other,
                },
            };
            match role {
                AxisRole::Time => (length, depth),
                AxisRole::Depth => (time, length),
                _ => (time, depth),
            }
        })
}

fn spatial_bounds(
    metadata: &DatasetMetadata,
    variable: &Variable,
) -> Option<(Bounds, usize, usize)> {
    if variable.dimensions.len() < 2 {
        return None;
    }
    let row_name = variable
        .dimensions
        .iter()
        .find(|name| {
            metadata
                .dimensions
                .iter()
                .find(|dimension| dimension.name == **name)
                .is_some_and(|dimension| dimension.role == AxisRole::Latitude)
        })
        .unwrap_or(&variable.dimensions[variable.dimensions.len() - 2]);
    let col_name = variable
        .dimensions
        .iter()
        .find(|name| {
            metadata
                .dimensions
                .iter()
                .find(|dimension| dimension.name == **name)
                .is_some_and(|dimension| dimension.role == AxisRole::Longitude)
        })
        .unwrap_or(&variable.dimensions[variable.dimensions.len() - 1]);
    if row_name == col_name {
        return None;
    }
    let rows = metadata
        .dimensions
        .iter()
        .find(|dimension| dimension.name == *row_name)?
        .length;
    let cols = metadata
        .dimensions
        .iter()
        .find(|dimension| dimension.name == *col_name)?
        .length;
    let (time, depth) = leading_lengths(metadata, variable);
    Some((Bounds::new(0, rows, 0, cols).ok()?, time, depth))
}

fn plane_bounds(
    metadata: &DatasetMetadata,
    variable: &Variable,
    x_axis: Option<&str>,
    y_axis: Option<&str>,
) -> Option<(Bounds, usize, usize)> {
    if data::is_mesh_variable(variable) {
        let (time, depth) = axis_lengths(metadata, variable);
        return Some((Bounds::new(0, 180, 0, 360).ok()?, time, depth));
    }
    let (Some(x_axis), Some(y_axis)) = (x_axis, y_axis) else {
        return spatial_bounds(metadata, variable);
    };
    let col_name = variable
        .dimensions
        .iter()
        .find(|name| name.eq_ignore_ascii_case(x_axis))?;
    let row_name = variable
        .dimensions
        .iter()
        .find(|name| name.eq_ignore_ascii_case(y_axis))?;
    if col_name == row_name {
        return None;
    }
    let rows = dimension_length(metadata, row_name)?;
    let cols = dimension_length(metadata, col_name)?;
    let (time, depth) = axis_lengths(metadata, variable);
    Some((Bounds::new(0, rows, 0, cols).ok()?, time, depth))
}

fn dimension_length(metadata: &DatasetMetadata, name: &str) -> Option<usize> {
    metadata
        .dimensions
        .iter()
        .find(|dimension| dimension.name.eq_ignore_ascii_case(name))
        .map(|dimension| dimension.length)
}

fn axis_lengths(metadata: &DatasetMetadata, variable: &Variable) -> (usize, usize) {
    let time = variable
        .dimensions
        .iter()
        .find_map(|name| {
            metadata
                .dimensions
                .iter()
                .find(|dimension| dimension.name.eq_ignore_ascii_case(name))
                .filter(|dimension| dimension.role == AxisRole::Time)
                .map(|dimension| dimension.length)
        })
        .unwrap_or(1);
    let depth = variable
        .dimensions
        .iter()
        .find_map(|name| {
            metadata
                .dimensions
                .iter()
                .find(|dimension| dimension.name.eq_ignore_ascii_case(name))
                .filter(|dimension| dimension.role == AxisRole::Depth)
                .map(|dimension| dimension.length)
        })
        .unwrap_or(1);
    (time, depth)
}

fn fixed_axes_for_plane(
    metadata: &DatasetMetadata,
    variable: &Variable,
    x_axis: Option<&str>,
    y_axis: Option<&str>,
    time_index: usize,
    depth_index: usize,
) -> Vec<(String, usize)> {
    let Some((x_axis, y_axis)) = x_axis.zip(y_axis) else {
        return Vec::new();
    };
    variable
        .dimensions
        .iter()
        .filter_map(|name| {
            if name.eq_ignore_ascii_case(x_axis) || name.eq_ignore_ascii_case(y_axis) {
                return None;
            }
            let dimension = metadata
                .dimensions
                .iter()
                .find(|dimension| dimension.name.eq_ignore_ascii_case(name));
            let length = dimension.map_or(1, |dimension| dimension.length);
            let index = match dimension.map(|dimension| dimension.role) {
                Some(AxisRole::Time) => time_index,
                Some(AxisRole::Depth) => depth_index,
                _ => 0,
            };
            Some((name.clone(), index.min(length.saturating_sub(1))))
        })
        .collect()
}

#[cfg(test)]
mod timeline_order_tests {
    use super::compare_time_labels;

    #[test]
    fn metadata_timestamps_sort_before_non_temporal_fallbacks() {
        assert_eq!(
            compare_time_labels("2026-09-09T12:00:00z", "2026-09-10T12:00:00z"),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            compare_time_labels("2026-09-10T12:00:00Z", "coordinate index"),
            std::cmp::Ordering::Less
        );
    }
}

#[cfg(test)]
mod sidebar_hit_tests {
    use super::{level_geometry, translate_scroll, translate_sidebar_position};
    use ncview_rs::app::{AppState, Command};
    use ncview_rs::data::{AxisRole, DatasetFormat, DatasetMetadata, Dimension, Variable};
    use ratatui::layout::Rect;

    const SIDEBAR: Rect = Rect::new(0, 0, 32, 30);

    fn metadata_with_variable() -> DatasetMetadata {
        DatasetMetadata {
            path: "f.nc".into(),
            format: DatasetFormat::NetCdf4,
            dimensions: vec![
                Dimension {
                    name: "lev".into(),
                    length: 12,
                    role: AxisRole::Depth,
                },
                Dimension {
                    name: "lat".into(),
                    length: 4,
                    role: AxisRole::Latitude,
                },
                Dimension {
                    name: "lon".into(),
                    length: 5,
                    role: AxisRole::Longitude,
                },
            ],
            variables: vec![Variable {
                name: "temp".into(),
                dimensions: vec!["lev".into(), "lat".into(), "lon".into()],
                numeric: true,
                units: None,
                long_name: None,
                standard_name: None,
            }],
        }
    }

    fn state_with_levels() -> AppState {
        let mut state = AppState::default();
        state.view.depth_length = 12;
        state.view.level_labels = (0..12).map(|i| format!("L{i}")).collect();
        state
    }

    #[test]
    fn stepper_and_level_rows_dispatch_depth_commands() {
        let state = state_with_levels();
        let metadata = metadata_with_variable();
        let section = level_geometry(SIDEBAR, &state.view, &metadata, "").unwrap();

        assert_eq!(
            translate_sidebar_position(
                section.stepper_prev.x + 1,
                section.stepper,
                SIDEBAR,
                &metadata,
                &state.view,
                "",
            ),
            Some(Command::MoveDepth(-1))
        );
        assert_eq!(
            translate_sidebar_position(
                section.stepper_next.x + 1,
                section.stepper,
                SIDEBAR,
                &metadata,
                &state.view,
                "",
            ),
            Some(Command::MoveDepth(1))
        );
        let index = section.window_top + 1;
        assert_eq!(
            translate_sidebar_position(
                section.list_rect.x + 1,
                section.list_top + 1,
                SIDEBAR,
                &metadata,
                &state.view,
                "",
            ),
            Some(Command::SetDepth(index))
        );
    }

    #[test]
    fn scroll_over_the_level_list_steps_depth_else_moves_variables() {
        let state = state_with_levels();
        let metadata = metadata_with_variable();
        let section = level_geometry(SIDEBAR, &state.view, &metadata, "").unwrap();
        assert_eq!(
            translate_scroll(
                SIDEBAR.x + 1,
                section.list_top,
                1,
                SIDEBAR,
                &metadata,
                &state.view,
                "",
            ),
            Some(Command::MoveDepth(1))
        );
        // Over the variable list (row 13): previous variable (index arg 0 = up).
        assert_eq!(
            translate_scroll(
                SIDEBAR.x + 1,
                SIDEBAR.y + 13,
                -1,
                SIDEBAR,
                &metadata,
                &state.view,
                "",
            ),
            Some(Command::SelectVariable(0))
        );
        // Outside the sidebar: ignored.
        assert_eq!(
            translate_scroll(
                SIDEBAR.x + 120,
                SIDEBAR.y + 13,
                -1,
                SIDEBAR,
                &metadata,
                &state.view,
                "",
            ),
            None
        );
    }

    #[test]
    fn hit_test_rows_match_the_rendered_widget() {
        let state = state_with_levels();
        let metadata = metadata_with_variable();
        let section = level_geometry(SIDEBAR, &state.view, &metadata, "").unwrap();
        // One plottable variable -> variable_rows = 1.
        // heading = first_variable_row(13) + 1 + separator(1) = 15.
        assert_eq!(section.heading, 15);
        assert_eq!(section.list_top, 18);
    }
}

#[cfg(test)]
mod level_bar_tests {
    use super::{depth_index_at, translate_mouse, translate_mouse_position};
    use ncview_rs::app::{AppState, Command};
    use ncview_rs::data::{DatasetFormat, DatasetMetadata};
    use ratatui::layout::Rect;

    const TERM: Rect = Rect::new(0, 0, 100, 30);
    // With show_level, the level band occupies rows 23..=25 (Task 1 test).
    const BAR_ROW: u16 = 24;

    fn empty_metadata() -> DatasetMetadata {
        DatasetMetadata {
            path: "f.nc".into(),
            format: DatasetFormat::NetCdf4,
            dimensions: Vec::new(),
            variables: Vec::new(),
        }
    }

    fn view_with_levels() -> AppState {
        let mut state = AppState::default();
        state.view.depth_length = 12;
        state
    }

    #[test]
    fn depth_index_scales_across_the_bar() {
        let bar = Rect::new(0, 23, 100, 3);
        assert_eq!(depth_index_at(1, bar, 12), 0);
        assert_eq!(depth_index_at(98, bar, 12), 11);
        assert_eq!(depth_index_at(50, bar, 12), 5);
        assert_eq!(depth_index_at(999, bar, 12), 11);
        assert_eq!(depth_index_at(0, bar, 1), 0);
    }

    #[test]
    fn click_on_the_bar_seeks_depth() {
        let state = view_with_levels();
        let metadata = empty_metadata();
        assert_eq!(
            translate_mouse_position(
                Command::MouseClick {
                    x: 1,
                    y: BAR_ROW,
                    right: false
                },
                TERM,
                &metadata,
                &state.view,
                "",
                None,
            ),
            Command::SetDepth(0)
        );
        assert_eq!(
            translate_mouse_position(
                Command::MouseClick {
                    x: 98,
                    y: BAR_ROW,
                    right: false
                },
                TERM,
                &metadata,
                &state.view,
                "",
                None,
            ),
            Command::SetDepth(11)
        );
    }

    #[test]
    fn press_on_the_bar_seeks_without_starting_a_canvas_drag() {
        let state = view_with_levels();
        assert_eq!(
            translate_mouse(
                Command::BeginDrag {
                    x: 50,
                    y: BAR_ROW,
                    zoom: false
                },
                TERM,
                &empty_metadata(),
                &state.view,
                "",
                None,
            ),
            Command::SetDepth(5)
        );
    }

    #[test]
    fn drag_over_the_bar_without_canvas_drag_seeks() {
        let state = view_with_levels();
        assert_eq!(
            translate_mouse(
                Command::UpdateDrag { x: 98, y: BAR_ROW },
                TERM,
                &empty_metadata(),
                &state.view,
                "",
                None,
            ),
            Command::SetDepth(11)
        );
    }

    #[test]
    fn wheel_over_the_bar_steps_depth() {
        let state = view_with_levels();
        assert_eq!(
            translate_mouse(
                Command::PointerScroll {
                    x: 50,
                    y: BAR_ROW,
                    delta: 1
                },
                TERM,
                &empty_metadata(),
                &state.view,
                "",
                None,
            ),
            Command::MoveDepth(1)
        );
    }

    #[test]
    fn absent_bar_leaves_events_untouched() {
        let state = AppState::default(); // depth_length == 1, no level band
        assert_eq!(
            translate_mouse_position(
                Command::MouseClick {
                    x: 50,
                    y: BAR_ROW,
                    right: false
                },
                TERM,
                &empty_metadata(),
                &state.view,
                "",
                None,
            ),
            Command::Pointer { x: 50, y: BAR_ROW }
        );
        assert_eq!(
            translate_mouse(
                Command::PointerScroll {
                    x: 50,
                    y: BAR_ROW,
                    delta: 1
                },
                TERM,
                &empty_metadata(),
                &state.view,
                "",
                None,
            ),
            Command::Pointer { x: 50, y: BAR_ROW }
        );
    }
}
