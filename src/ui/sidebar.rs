use ratatui::{
    Frame,
    layout::Rect,
    style::Style,
    text::{Line, Span},
    widgets::Paragraph,
};

use crate::data::{DatasetMetadata, Variable};
use crate::ui::level::{LevelPanel, level_section, level_window};
use crate::{
    app::ColorScaleScope,
    render::colors::{Palette, ScaleMode},
};

use super::theme;

fn truncate_text(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    if max_chars <= 1 {
        return "…".chars().take(max_chars).collect();
    }
    let mut result = value.chars().take(max_chars - 1).collect::<String>();
    result.push('…');
    result
}

fn truncate_variable_name(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    if max_chars <= 1 {
        return "…".chars().take(max_chars).collect();
    }

    let characters = value.chars().collect::<Vec<_>>();
    let suffix_len = (max_chars - 1) / 2;
    let prefix_len = max_chars - 1 - suffix_len;
    let prefix = characters[..prefix_len].iter().collect::<String>();
    let suffix = characters[characters.len() - suffix_len..]
        .iter()
        .collect::<String>();
    format!("{prefix}…{suffix}")
}

fn truncate_path(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    if max_chars <= 1 {
        return "…".chars().take(max_chars).collect();
    }
    let tail = value
        .rsplit_once('/')
        .map_or(value, |(_, filename)| filename);
    format!("…/{}", truncate_text(tail, max_chars.saturating_sub(2)))
}

pub fn filter_variables<'a>(variables: &'a [Variable], query: &str) -> Vec<&'a Variable> {
    let query = query.to_ascii_lowercase();
    if query.is_empty() {
        return variables.iter().collect();
    }
    variables
        .iter()
        .filter(|variable| {
            let name = variable.name.to_ascii_lowercase();
            let mut chars = name.chars();
            query
                .chars()
                .all(|needle| chars.by_ref().any(|candidate| candidate == needle))
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub fn render(
    frame: &mut Frame,
    area: Rect,
    filename: &str,
    metadata: &DatasetMetadata,
    selected_variable: Option<&str>,
    palette: &Palette,
    limits: Option<(f64, f64)>,
    filter_range: Option<(f64, f64)>,
    scale: ScaleMode,
    color_scale_scope: ColorScaleScope,
) {
    render_with_search(
        frame,
        area,
        filename,
        metadata,
        selected_variable,
        palette,
        limits,
        filter_range,
        scale,
        color_scale_scope,
        "",
        false,
        None,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn render_with_search(
    frame: &mut Frame,
    area: Rect,
    filename: &str,
    metadata: &DatasetMetadata,
    selected_variable: Option<&str>,
    palette: &Palette,
    limits: Option<(f64, f64)>,
    filter_range: Option<(f64, f64)>,
    scale: ScaleMode,
    color_scale_scope: ColorScaleScope,
    variable_query: &str,
    variable_search_active: bool,
    level: Option<LevelPanel<'_>>,
) {
    let content_width = usize::from(area.width.saturating_sub(2));
    let limit_text = limits.map_or_else(
        || "limits: auto".to_string(),
        |(min, max)| format!("limits: {min:.4}..{max:.4}"),
    );
    let plottable = metadata
        .variables
        .iter()
        .filter(|variable| {
            variable.numeric
                && (crate::data::is_mesh_variable(variable) || variable.dimensions.len() >= 2)
        })
        .cloned()
        .collect::<Vec<_>>();
    let variables = filter_variables(&plottable, variable_query)
        .into_iter()
        .take(8)
        .map(|variable| {
            let marker = if selected_variable == Some(variable.name.as_str()) {
                "▶"
            } else {
                " "
            };
            (marker.to_string(), variable.name.clone())
        })
        .collect::<Vec<_>>();
    let dimensions = metadata.dimensions.iter().take(8).collect::<Vec<_>>();
    let mut lines = vec![Line::from(vec![
        Span::styled(
            format!("{} ", theme::ICON_FILE),
            theme::title_style(theme::BLUE),
        ),
        Span::styled(
            truncate_path(filename, content_width.saturating_sub(2)),
            theme::muted_style(),
        ),
    ])];
    lines.push(Line::from(Span::styled(
        format!("{}  Colormap", theme::ICON_PALETTE),
        theme::title_style(theme::PEACH),
    )));
    let palette_label = if palette.is_reversed() {
        format!("{} (reversed)", palette.name())
    } else {
        palette.name().to_string()
    };
    lines.push(Line::from(vec![
        Span::styled("  ● ", Style::default().fg(theme::PEACH)),
        Span::styled(
            truncate_text(&palette_label, content_width.saturating_sub(4)),
            Style::default().fg(theme::TEXT),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::styled(format!("  scale: {}  ", scale.name()), theme::muted_style()),
        Span::styled("[s]", theme::title_style(theme::TEAL)),
    ]));
    lines.push(Line::from(Span::styled(
        "  View actions",
        theme::title_style(theme::TEAL),
    )));
    let button = |label: &str, color| {
        Span::styled(
            label.to_string(),
            Style::default().fg(theme::TEXT).bg(color),
        )
    };
    lines.push(Line::from(vec![
        button("[c] MAP", theme::SURFACE_ALT),
        Span::raw(" "),
        button("[v] REV", theme::SURFACE_ALT),
        Span::raw(" "),
        button("[a] AUTO", theme::SURFACE_ALT),
    ]));
    lines.push(Line::from(vec![
        button("[l] LIM", theme::SURFACE_ALT),
        Span::raw(" "),
        button("[f]MASK", theme::SURFACE_ALT),
        Span::raw(" "),
        button("[x] AXES", theme::SURFACE_ALT),
    ]));
    lines.push(Line::from(Span::styled(
        "  Navigation",
        theme::title_style(theme::BLUE),
    )));
    lines.push(Line::from(vec![
        button("[r] RESET", theme::SURFACE_ALT),
        Span::raw(" "),
        button("[<] DATE", theme::SURFACE_ALT),
        Span::raw(" "),
        button("[>] DATE", theme::SURFACE_ALT),
    ]));
    lines.push(Line::from(vec![
        button("[-]SPD", theme::SURFACE_ALT),
        Span::raw(" "),
        button("[+]SPD", theme::SURFACE_ALT),
        Span::raw(" "),
        button("[z]SCALE", theme::SURFACE_ALT),
    ]));
    lines.push(Line::from(Span::styled(
        format!("{}  Variables", theme::ICON_VARIABLE),
        theme::title_style(theme::MAUVE),
    )));
    lines.push(Line::from(vec![
        Span::styled(
            "  / ",
            theme::title_style(if variable_search_active {
                theme::TEAL
            } else {
                theme::SURFACE_ALT
            }),
        ),
        Span::styled(
            if variable_query.is_empty() {
                "type to search…".to_string()
            } else {
                truncate_text(variable_query, content_width.saturating_sub(4))
            },
            if variable_search_active {
                Style::default().fg(theme::TEXT)
            } else {
                theme::muted_style()
            },
        ),
    ]));
    let variables_top = lines.len();
    if variables.is_empty() {
        lines.push(Line::from(Span::styled(
            "  no plottable fields",
            theme::muted_style(),
        )));
    } else {
        for (marker, name) in variables {
            let selected = marker == "▶";
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  {marker} "),
                    theme::title_style(if selected {
                        theme::TEAL
                    } else {
                        theme::SURFACE_ALT
                    }),
                ),
                Span::styled(
                    truncate_variable_name(&name, content_width.saturating_sub(4)),
                    Style::default().fg(if selected {
                        theme::TEXT
                    } else {
                        theme::SUBTEXT
                    }),
                ),
            ]));
        }
    }
    let variable_rows = lines.len() - variables_top;
    if let Some(panel) = level {
        let stepper_span = (content_width / 2) as u16;
        if let Some(base) = level_section(area, variable_rows, panel.labels.len(), stepper_span) {
            let (top, len) = level_window(panel.labels.len(), base.list_rows, panel.cursor);
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "  Level",
                theme::title_style(if panel.focused {
                    theme::TEAL
                } else {
                    theme::BLUE
                }),
            )));
            let readout = panel
                .labels
                .get(panel.selected)
                .cloned()
                .unwrap_or_else(|| format!("index {}", panel.selected));
            lines.push(Line::from(Span::styled(
                truncate_text(
                    &format!(
                        "  {readout} [{} / {}]",
                        panel.selected + 1,
                        panel.labels.len()
                    ),
                    content_width,
                ),
                Style::default().fg(theme::TEXT),
            )));
            lines.push(Line::from(vec![
                button("◂ Prev", theme::SURFACE_ALT),
                Span::raw("  "),
                button("Next ▸", theme::SURFACE_ALT),
            ]));
            for index in top..top + len {
                let label = panel
                    .labels
                    .get(index)
                    .map_or_else(|| format!("index {index}"), String::clone);
                let is_selected = index == panel.selected;
                let is_cursor = panel.focused && index == panel.cursor;
                let marker = if is_cursor {
                    "▸"
                } else if is_selected {
                    "●"
                } else {
                    " "
                };
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("  {marker} "),
                        theme::title_style(if is_cursor || is_selected {
                            theme::TEAL
                        } else {
                            theme::SURFACE_ALT
                        }),
                    ),
                    Span::styled(
                        truncate_text(&label, content_width.saturating_sub(4)),
                        Style::default().fg(if is_selected {
                            theme::TEXT
                        } else {
                            theme::SUBTEXT
                        }),
                    ),
                ]));
            }
        }
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        format!("{}  Dimensions", theme::ICON_DIMENSION),
        theme::title_style(theme::BLUE),
    )));
    for dimension in dimensions {
        let label = format!("  • {} = {}", dimension.name, dimension.length);
        lines.push(Line::from(Span::styled(
            truncate_text(&label, content_width),
            theme::muted_style(),
        )));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("  ↕ ", Style::default().fg(theme::TEAL)),
        Span::styled(
            truncate_text(&limit_text, content_width.saturating_sub(4)),
            theme::muted_style(),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::styled("  ▪ ", Style::default().fg(theme::RED)),
        Span::styled(
            truncate_text(
                &filter_range.map_or_else(
                    || "mask: off".into(),
                    |(min, max)| format!("mask: {min:.4}..{max:.4}"),
                ),
                content_width.saturating_sub(4),
            ),
            theme::muted_style(),
        ),
    ]));
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        format!("  scope: {}  [z]", color_scale_scope.name(),),
        theme::muted_style(),
    )));
    lines.push(Line::from(Span::styled(
        "  [v] reverse  [b] map backdrop",
        theme::muted_style(),
    )));
    frame.render_widget(
        Paragraph::new(lines).block(theme::panel("Dataset", theme::BLUE)),
        area,
    );
}

#[cfg(test)]
mod tests {
    use super::truncate_variable_name;

    fn empty_metadata() -> crate::data::DatasetMetadata {
        crate::data::DatasetMetadata {
            path: "f.nc".into(),
            format: crate::data::DatasetFormat::NetCdf4,
            dimensions: Vec::new(),
            variables: Vec::new(),
        }
    }

    fn render_sidebar(level: Option<crate::ui::level::LevelPanel<'_>>) -> String {
        use ratatui::{Terminal, backend::TestBackend};
        let mut terminal = Terminal::new(TestBackend::new(32, 30)).unwrap();
        terminal
            .draw(|frame| {
                super::render_with_search(
                    frame,
                    frame.area(),
                    "f.nc",
                    &empty_metadata(),
                    None,
                    &crate::render::colors::Palette::Viridis,
                    None,
                    None,
                    crate::render::colors::ScaleMode::Linear,
                    crate::app::ColorScaleScope::CurrentView,
                    "",
                    false,
                    level,
                )
            })
            .unwrap();
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect()
    }

    #[test]
    fn level_section_renders_header_stepper_and_list() {
        use crate::ui::level::LevelPanel;
        let labels: Vec<String> = (0..6).map(|i| format!("L{i}")).collect();
        let rendered = render_sidebar(Some(LevelPanel {
            labels: &labels,
            selected: 2,
            cursor: 2,
            focused: false,
        }));
        assert!(rendered.contains("Level"), "{rendered}");
        assert!(rendered.contains("L2 [3 / 6]"), "{rendered}");
        assert!(rendered.contains("Prev"), "{rendered}");
        assert!(rendered.contains("Next"), "{rendered}");
    }

    #[test]
    fn level_section_is_absent_without_levels() {
        let rendered = render_sidebar(None);
        assert!(!rendered.contains("Level"), "{rendered}");
    }

    #[test]
    fn variable_truncation_preserves_both_ends() {
        let name = "dust_dry_gt0um_aerosol_optical_thickness_545nm-565nm";
        let displayed = truncate_variable_name(name, 26);

        assert_eq!(displayed.chars().count(), 26);
        assert!(displayed.starts_with("dust_dry_gt0"));
        assert!(displayed.ends_with("545nm-565nm"));
        assert!(displayed.contains('…'));
    }

    #[test]
    fn short_variable_names_are_unchanged() {
        assert_eq!(truncate_variable_name("temperature", 26), "temperature");
    }
}
