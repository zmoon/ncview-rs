use super::chart;
use crate::app::{
    AxisField, COMMAND_PALETTE, LimitField, Overlay, PlotAxisField, PlotKind, PlotXAxis, PlotYAxis,
    ViewModel, palette_matches,
};
use crate::data::{DatasetMetadata, Variable};
use crate::render::protocol::GraphicsRenderer;
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::Style,
    text::{Line, Span},
    widgets::{Block, Clear, Paragraph, Wrap},
};

use super::{sidebar, theme};

fn popup_panel<'a>(title: &'a str, accent: ratatui::style::Color) -> Block<'a> {
    theme::panel(title, accent).title_top(theme::close_button())
}

pub fn render(
    frame: &mut Frame,
    area: Rect,
    view: &ViewModel,
    metadata: &DatasetMetadata,
    variable_query: &str,
    chart_graphics: Option<&mut GraphicsRenderer>,
) {
    if view.variable_search_active {
        render_variable_browser(frame, area, view, metadata, variable_query);
        return;
    }
    let Some(overlay) = view.overlay else { return };
    let title = match overlay {
        Overlay::Limits => "Limits",
        Overlay::Filter => "Data filter",
        Overlay::Axis => "Axes",
        Overlay::TimeSeries => "Time series",
        Overlay::Plot => "Plot",
        Overlay::CommandPalette => "Command Palette",
    };
    let message = match overlay {
        Overlay::Limits => "Type to replace the selected value; Tab switches fields",
        Overlay::Filter => "Values outside the range are masked",
        Overlay::Axis => "Choose distinct X and Y dimensions",
        Overlay::TimeSeries => "Values across the time dimension",
        Overlay::Plot => "Choose a plot and its axes",
        Overlay::CommandPalette => "Type to filter commands; Enter runs the selected action",
    };
    let width = if matches!(overlay, Overlay::CommandPalette | Overlay::Plot) {
        area.width.saturating_mul(3) / 4
    } else {
        area.width.saturating_mul(3) / 5
    };
    let height = if matches!(overlay, Overlay::CommandPalette | Overlay::Plot) {
        area.height.saturating_mul(3) / 5
    } else {
        area.height.saturating_mul(2) / 5
    };
    let popup = Rect {
        x: area.x + (area.width - width) / 2,
        y: area.y + (area.height - height) / 2,
        width,
        height,
    };
    let shadow = Rect {
        x: popup.x.saturating_add(1),
        y: popup.y.saturating_add(1),
        width: popup.width,
        height: popup.height,
    };
    frame.render_widget(Clear, popup);
    frame.render_widget(
        ratatui::widgets::Block::default().style(Style::default().bg(theme::SHADOW)),
        shadow,
    );
    if matches!(overlay, Overlay::CommandPalette) {
        let matches = palette_matches(&view.palette_query);
        let mut lines = vec![format!("Search: {}", view.palette_query)];
        lines.push(String::new());
        if matches.is_empty() {
            lines.push("No matching commands".into());
        } else {
            for (position, index) in matches.iter().enumerate() {
                let entry = &COMMAND_PALETTE[*index];
                let marker = if position == view.palette_index {
                    ">"
                } else {
                    " "
                };
                lines.push(format!("{marker} {:<34} {}", entry.label, entry.shortcut));
            }
        }
        lines.push(String::new());
        lines.push("↑↓ select   Enter run   Esc close".into());
        frame.render_widget(
            Paragraph::new(lines.join("\n")).block(popup_panel(title, theme::MAUVE)),
            popup,
        );
    } else if matches!(overlay, Overlay::Limits | Overlay::Filter) {
        let draft = view.limit_draft.as_ref();
        let min = draft.map_or("".to_string(), |draft| draft.min.clone());
        let max = draft.map_or("".to_string(), |draft| draft.max.clone());
        let active = draft.map(|draft| draft.active);
        let text = format!(
            "{}: [ {}{} ]\n{}: [ {}{} ]\n\nType replaces value   Backspace edits\nTab: switch field   Enter: apply   Esc: cancel",
            if matches!(overlay, Overlay::Filter) {
                "Keep from"
            } else {
                "Minimum"
            },
            if active == Some(LimitField::Min) {
                "> "
            } else {
                "  "
            },
            min,
            if matches!(overlay, Overlay::Filter) {
                "Keep through"
            } else {
                "Maximum"
            },
            if active == Some(LimitField::Max) {
                "> "
            } else {
                "  "
            },
            max
        );
        frame.render_widget(
            Paragraph::new(text).block(popup_panel(title, theme::PEACH)),
            popup,
        );
    } else if matches!(overlay, Overlay::Axis) {
        let draft = view.axis_draft.as_ref();
        let x = draft.map_or("".to_string(), |draft| draft.x.clone());
        let y = draft.map_or("".to_string(), |draft| draft.y.clone());
        let active = draft.map(|draft| draft.active);
        let options = if view.axis_options.is_empty() {
            metadata
                .variables
                .iter()
                .find(|variable| view.selected_variable.as_deref() == Some(variable.name.as_str()))
                .map(|variable| variable.dimensions.join(", "))
                .unwrap_or_else(|| "no dimensions discovered".into())
        } else {
            view.axis_options.join(", ")
        };
        let text = format!(
            "X axis: [ {}{} ]\nY axis: [ {}{} ]\n\nAvailable: {options}\n↑↓ choose axis   Tab: switch field\nType to replace value   Enter: apply   Esc: cancel",
            if active == Some(AxisField::X) {
                "> "
            } else {
                "  "
            },
            x,
            if active == Some(AxisField::Y) {
                "> "
            } else {
                "  "
            },
            y
        );
        frame.render_widget(
            Paragraph::new(text).block(popup_panel(title, theme::BLUE)),
            popup,
        );
    } else if matches!(overlay, Overlay::Plot) {
        render_plot(frame, popup, view, chart_graphics);
    } else if matches!(overlay, Overlay::TimeSeries) {
        let series = plot_series_for_view(view);
        chart::render_plot(
            frame,
            popup,
            PlotKind::TimeSeries,
            PlotXAxis::ValidTime,
            PlotYAxis::Value,
            None,
            &series,
            &[],
            chart_graphics,
        );
    } else {
        frame.render_widget(
            Paragraph::new(message).block(theme::panel(title, theme::MAUVE)),
            popup,
        );
    }
}

fn render_plot(
    frame: &mut Frame,
    popup: Rect,
    view: &ViewModel,
    chart_graphics: Option<&mut GraphicsRenderer>,
) {
    let block = popup_panel("Plot", theme::MAUVE);
    let inner = block.inner(popup);
    frame.render_widget(block, popup);
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(7), Constraint::Min(5)])
        .split(inner);
    let draft = view.plot_draft;
    let x_label = match draft.x_axis {
        PlotXAxis::ValidTime => "valid time",
        PlotXAxis::SampleIndex => "sample index",
        PlotXAxis::Longitude => "longitude",
        PlotXAxis::Latitude => "latitude",
        PlotXAxis::Dimension(_index) if draft.kind == PlotKind::VerticalProfile => "value",
        PlotXAxis::Dimension(index) => view
            .axis_options
            .get(index)
            .map(String::as_str)
            .unwrap_or("dimension"),
        PlotXAxis::Value => "value",
    };
    let y_label = match draft.y_axis {
        PlotYAxis::Value if draft.kind == PlotKind::VerticalProfile => view
            .axis_options
            .get(match draft.x_axis {
                PlotXAxis::Dimension(index) => index,
                _ => usize::MAX,
            })
            .map(String::as_str)
            .unwrap_or("level"),
        PlotYAxis::Value => "value",
        PlotYAxis::Frequency => "frequency",
        PlotYAxis::Density => "density (%)",
    };
    let kind_label = match draft.kind {
        PlotKind::TimeSeries => "time series",
        PlotKind::Scatter => "scatter",
        PlotKind::Histogram => "histogram",
        PlotKind::Cdf => "CDF",
        PlotKind::VerticalProfile => "vertical profile",
    };
    let x_marker = if draft.active == PlotAxisField::X {
        ">"
    } else {
        " "
    };
    let y_marker = if draft.active == PlotAxisField::Y {
        ">"
    } else {
        " "
    };
    let selected_count = view
        .selected_points
        .len()
        .max(usize::from(view.selected_point.is_some()));
    let target_label = if selected_count == 0 {
        "view domain summary".to_string()
    } else if selected_count == 1
        && let (Some(lat), Some(lon)) = (
            view.selected_coordinates.latitude,
            view.selected_coordinates.longitude,
        )
    {
        format!("point at lat={lat:.2}°, lon={lon:.2}°")
    } else {
        format!("{selected_count} selected point(s)")
    };
    let controls = format!(
        "type: [t] time series  [d] scatter  [h] histogram  [k] CDF  [u] profile   (current: {kind_label})\n\
target: {target_label}\n\
{x_marker} X axis: {x_label}\n\
{y_marker} Y axis: {y_label}\n\
Tab switches axes  •  ↑↓/←→ changes the selected axis  •  m adds/removes points",
    );
    frame.render_widget(
        Paragraph::new(controls).block(theme::panel("Plot controls", theme::BLUE)),
        rows[0],
    );
    let series = plot_series_for_view(view);
    let histogram_values =
        if !view.selected_points.is_empty() && series.iter().any(|item| item.data.len() > 1) {
            series
                .iter()
                .flat_map(|item| item.data.iter().map(|(_, value)| *value))
                .collect::<Vec<_>>()
        } else {
            view.slice
                .as_ref()
                .map(|slice| slice.values.iter().copied().collect::<Vec<_>>())
                .unwrap_or_default()
        };
    chart::render_plot(
        frame,
        rows[1],
        draft.kind,
        draft.x_axis,
        draft.y_axis,
        match draft.x_axis {
            PlotXAxis::Dimension(index) => view.axis_options.get(index).map(String::as_str),
            _ => None,
        },
        &series,
        &histogram_values,
        chart_graphics,
    );
}

fn plot_series_for_view(view: &ViewModel) -> Vec<crate::app::PlotSeries> {
    if view.plot_series.is_empty() {
        vec![crate::app::PlotSeries {
            point: view.selected_point.unwrap_or((0, 0)),
            label: "selected point".into(),
            data: view.time_series.clone(),
            labels: view.time_series_labels.clone(),
        }]
    } else {
        view.plot_series.clone()
    }
}

fn render_variable_browser(
    frame: &mut Frame,
    area: Rect,
    view: &ViewModel,
    metadata: &DatasetMetadata,
    variable_query: &str,
) {
    let width = area.width.saturating_mul(4).saturating_div(5).max(1);
    let height = area.height.saturating_mul(4).saturating_div(5).max(1);
    let popup = Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(height) / 2,
        width: width.min(area.width),
        height: height.min(area.height),
    };
    let shadow = Rect {
        x: popup.x.saturating_add(1),
        y: popup.y.saturating_add(1),
        width: popup.width,
        height: popup.height,
    };
    frame.render_widget(Clear, popup);
    frame.render_widget(
        ratatui::widgets::Block::default().style(Style::default().bg(theme::SHADOW)),
        shadow,
    );

    let block = popup_panel("Variables", theme::MAUVE);
    let inner = block.inner(popup);
    let plottable = metadata
        .variables
        .iter()
        .filter(|variable| {
            variable.numeric
                && (crate::data::is_mesh_variable(variable) || variable.dimensions.len() >= 2)
        })
        .cloned()
        .collect::<Vec<Variable>>();
    let filtered = sidebar::filter_variables(&plottable, variable_query);
    let selected = view
        .variable_browser_index
        .min(filtered.len().saturating_sub(1));
    let name_width = usize::from(inner.width.saturating_sub(3)).max(1);
    let entries = filtered
        .iter()
        .enumerate()
        .map(|(index, variable)| {
            wrap_name(&variable.name, name_width)
                .into_iter()
                .enumerate()
                .map(|(line_index, chunk)| {
                    let marker = if line_index == 0 {
                        if index == selected { "▶ " } else { "  " }
                    } else {
                        "  "
                    };
                    let style = if index == selected {
                        theme::title_style(theme::TEXT)
                    } else {
                        theme::muted_style()
                    };
                    Line::from(vec![
                        Span::styled(marker, style),
                        Span::styled(chunk, style),
                    ])
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let list_height = usize::from(inner.height).saturating_sub(4);
    let start = visible_window_start(&entries, selected, list_height);
    let mut lines = vec![Line::from(vec![
        Span::styled("Search: ", theme::title_style(theme::TEAL)),
        Span::styled(
            if variable_query.is_empty() {
                "<all>"
            } else {
                variable_query
            },
            theme::muted_style(),
        ),
    ])];
    lines.push(Line::from(""));
    if filtered.is_empty() {
        lines.push(Line::from(Span::styled(
            "No matching plottable variables",
            theme::muted_style(),
        )));
    } else {
        let mut used = 0usize;
        for entry in entries.iter().skip(start) {
            if used >= list_height {
                break;
            }
            let room = list_height - used;
            lines.extend(entry.iter().take(room).cloned());
            used += entry.len().min(room);
        }
    }
    lines.push(Line::from(Span::styled(
        format!(
            "{} field(s)  ↑↓ select  Enter open  Esc close",
            filtered.len()
        ),
        theme::muted_style(),
    )));
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(block),
        popup,
    );
}

fn wrap_name(value: &str, width: usize) -> Vec<String> {
    let characters = value.chars().collect::<Vec<_>>();
    if characters.is_empty() {
        return vec![String::new()];
    }
    characters
        .chunks(width.max(1))
        .map(|chunk| chunk.iter().collect())
        .collect()
}

fn visible_window_start(entries: &[Vec<Line<'_>>], selected: usize, height: usize) -> usize {
    if entries.is_empty() || height == 0 {
        return 0;
    }
    let selected = selected.min(entries.len() - 1);
    let mut start = selected;
    let mut used = entries[selected].len();
    while start > 0 && used + entries[start - 1].len() <= height {
        start -= 1;
        used += entries[start].len();
    }
    start
}
