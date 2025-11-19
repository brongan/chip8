#![feature(iter_array_chunks)]
use eframe::egui::{
    self, Align, Button, Color32, Frame, Key, Layout, RichText, ScrollArea, SidePanel, TextStyle,
    Vec2,
};
use rodio::{OutputStream, Sink, Source};
use spin_sleep::sleep;
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64};
use std::sync::mpsc::{Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};
use strum::IntoEnumIterator;

use chip8::{CPU, Emulator, Keypad, Register, Screen};

/// Renders the 64x32 screen state into a displayable image
fn render_screen(screen: &Screen, on_color: Color32, off_color: Color32) -> egui::ColorImage {
    let pixels: Vec<Color32> = screen
        .0
        .iter()
        .flat_map(|row| row.iter())
        .map(|pixel| if *pixel { on_color } else { off_color })
        .collect();

    egui::ColorImage {
        size: [64, 32],
        pixels,
        source_size: Vec2::default(),
    }
}

impl DebuggerApp {
    fn new(cc: &eframe::CreationContext, rom: Option<Vec<u8>>) -> Self {
        let (state_tx, state_rx) = std::sync::mpsc::sync_channel(1);
        let (rom_tx, rom_rx) = std::sync::mpsc::channel::<Vec<u8>>();
        let keypad = Keypad::default();
        let ctx = cc.egui_ctx.clone();
        ctx.set_visuals(egui::Visuals::dark());

        let running = Arc::new(AtomicBool::new(rom.is_some()));
        let target_ips = Arc::new(AtomicU32::new(700));
        let target_fps = Arc::new(AtomicU32::new(60));
        let instruction_counter: Arc<AtomicU64> = Arc::default();

        let cpu = CPU::new(rom, keypad.clone());
        let mut emulator = Emulator {
            cpu: cpu.clone(),
            target_fps: target_fps.clone(),
            cycle_accumulator: 0,
        };
        let ips = target_ips.clone();
        let counter = instruction_counter.clone();
        let running_clone = running.clone();
        thread::spawn(move || {
            loop {
                if !running_clone.load(Relaxed) {
                    sleep(Duration::from_millis(250));
                    continue;
                }

                if let Ok(new_rom) = rom_rx.try_recv() {
                    emulator.load_rom(new_rom);
                    if state_tx.send(emulator.cpu.clone()).is_err() {
                        eprintln!("UI closed channel after ROM load.");
                        break;
                    }
                }

                let start = Instant::now();
                let ips = ips.load(Relaxed);
                let instruction_time = Duration::from_secs_f64(1.0 / ips as f64);
                counter.fetch_add(1, Relaxed);
                if let Some(state) = emulator.tick(ips) {
                    if state_tx.send(state).is_err() {
                        eprintln!("UI closed channel.");
                        break;
                    }
                    ctx.request_repaint();
                }

                let elapsed = start.elapsed();
                if elapsed < instruction_time {
                    sleep(instruction_time - elapsed);
                }
            }
        });

        let on_pixel_color = Color32::WHITE;
        let off_pixel_color = Color32::BLACK;
        let image = egui::ColorImage::new([64, 32], vec![Color32::BLACK; 64 * 32]);
        let display_texture = cc
            .egui_ctx
            .load_texture("LCD", image, egui::TextureOptions::NEAREST);

        let stream =
            rodio::OutputStreamBuilder::open_default_stream().expect("open default audio stream");
        let sink = rodio::Sink::connect_new(&stream.mixer());
        let beep_sound = rodio::source::SineWave::new(440.0) // A 440hz tone
            .amplify(0.20);
        sink.append(beep_sound);
        sink.pause();

        Self {
            rom_tx,
            state_rx,
            last_state: cpu,
            keypad,
            display_texture,
            _stream: stream,
            sink,
            on_pixel_color,
            off_pixel_color,
            game_scale: 16.0,
            target_ips,
            target_fps,
            running,
            last_frame: Instant::now(),
            instruction_counter,
        }
    }

    fn check_for_updates(&mut self) {
        if let Ok(cpu) = self.state_rx.try_recv() {
            if cpu.is_beep() {
                self.sink.play();
            } else {
                self.sink.pause();
            }
            self.last_state = cpu;
        }
    }

    /// Check for and map CHIP-8 key presses
    fn check_keyboard(ctx: &egui::Context) -> u16 {
        const KEY_MAP: &[(Key, u16)] = &[
            (Key::Num1, 1 << 0x1),
            (Key::Num2, 1 << 0x2),
            (Key::Num3, 1 << 0x3),
            (Key::Num4, 1 << 0xC),
            (Key::Q, 1 << 0x4),
            (Key::W, 1 << 0x5),
            (Key::E, 1 << 0x6),
            (Key::R, 1 << 0xD),
            (Key::A, 1 << 0x7),
            (Key::S, 1 << 0x8),
            (Key::D, 1 << 0x9),
            (Key::F, 1 << 0xE),
            (Key::Z, 1 << 0xA),
            (Key::X, 1 << 0x0),
            (Key::C, 1 << 0xB),
            (Key::V, 1 << 0xF),
        ];

        ctx.input(|i| {
            KEY_MAP.iter().fold(0u16, |mut keypad, (key, mask)| {
                if i.key_down(*key) {
                    keypad |= mask;
                }
                keypad
            })
        })
    }

    /// Renders the 16 V-registers
    fn render_register_viewer(&self, ui: &mut egui::Ui, cpu: &CPU) {
        egui::CollapsingHeader::new("Register Viewer")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("register_grid")
                    .num_columns(4)
                    .spacing([10.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        for row in Register::iter().array_chunks::<2>() {
                            for reg in row {
                                ui.label(
                                    RichText::new(format!("{reg}"))
                                        .text_style(TextStyle::Monospace),
                                );
                                ui.label(
                                    RichText::new(format!("0x{:02X}", cpu.get_register(reg)))
                                        .text_style(TextStyle::Monospace),
                                );
                            }
                            ui.end_row();
                        }
                    });

                ui.separator();

                egui::Grid::new("special_registers")
                    .num_columns(2)
                    .spacing([10.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        ui.style_mut().override_text_style = Some(TextStyle::Monospace);
                        ui.spacing_mut().item_spacing.x = 2.0;
                        ui.label(RichText::new("PC"));
                        ui.label(RichText::new(format!("0x{:04X}", cpu.get_pc())));

                        ui.end_row();

                        ui.label(RichText::new("IR"));
                        ui.label(RichText::new(format!("0x{:04X}", cpu.fetch())));
                        ui.end_row();

                        ui.label(RichText::new("I"));
                        ui.label(RichText::new(format!("0x{:04X}", cpu.get_index())));
                        ui.end_row();

                        ui.label(RichText::new("Delay"));
                        ui.label(RichText::new(format!("{}", cpu.get_delay_timer())));
                        ui.end_row();

                        ui.label(RichText::new("Sound"));
                        ui.label(RichText::new(format!("{}", cpu.get_sound_timer())));
                        ui.end_row();
                    });
            });
    }

    /// Renders the CPU stack
    fn render_stack_viewer(&self, ui: &mut egui::Ui, cpu: &CPU) {
        egui::CollapsingHeader::new("Stack Viewer")
            .default_open(true)
            .show(ui, |ui| {
                ScrollArea::vertical().max_height(150.0).show(ui, |ui| {
                    egui::Grid::new("stack_grid")
                        .num_columns(2)
                        .striped(true)
                        .show(ui, |ui| {
                            ui.label(RichText::new("Depth").strong());
                            ui.label(RichText::new("Contents").strong());
                            ui.end_row();

                            for (i, &addr) in cpu.get_stack().iter().enumerate().rev() {
                                ui.label(format!("{}", i));
                                ui.label(
                                    RichText::new(format!("0x{:04X}", addr))
                                        .text_style(TextStyle::Monospace),
                                );
                                ui.end_row();
                            }
                        });
                });
            });
    }

    fn render_memory_viewer(&self, ui: &mut egui::Ui, cpu: &CPU) {
        egui::CollapsingHeader::new("Memory Viewer")
            .default_open(true)
            .show(ui, |ui| {
                ScrollArea::vertical().show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.style_mut().override_text_style = Some(TextStyle::Monospace);
                        ui.spacing_mut().item_spacing.x = 2.0;

                        for (i, chunk) in cpu.get_memory().chunks(16).enumerate() {
                            let addr = i * 16;
                            let color = if (addr..addr + 16).contains(&(cpu.get_pc() as usize)) {
                                Color32::YELLOW
                            } else {
                                Color32::LIGHT_GREEN
                            };
                            ui.colored_label(color, format!("0x{:04X}", addr));

                            ui.add(egui::Separator::default().vertical().shrink(10.0));
                            for byte in chunk {
                                ui.colored_label(color, format!("{:02X}", byte));
                            }
                            ui.add(egui::Separator::default().vertical().shrink(10.0));
                            ui.label(
                                RichText::new(
                                    chunk
                                        .iter()
                                        .map(|&b| match b {
                                            b' '..=b'~' => b as char,
                                            _ => '.',
                                        })
                                        .collect::<String>(),
                                )
                                .color(color),
                            );
                            ui.end_row();
                        }
                    });
                });
            });
    }

    fn render_settings_panel(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Settings Menu")
            .default_open(true)
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing = egui::vec2(2.0, 2.0);

                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 8.0;
                    let running = self.running.load(Relaxed);

                    let run_button = Button::new(if running { "Pause" } else { "Run" })
                        .min_size(egui::vec2(60.0, 30.0));
                    let load_button = Button::new("Load ROM...").min_size(egui::vec2(150.0, 30.0));

                    if ui.add(run_button).clicked() {
                        self.running.store(!running, Relaxed);
                    }

                    if ui.add(load_button).clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("CHIP-8 ROM", &["ch8", "rom"])
                            .add_filter("All Files", &["*"])
                            .pick_file()
                        {
                            match std::fs::read(&path) {
                                Ok(rom_data) => {
                                    if let Err(e) = self.rom_tx.send(rom_data) {
                                        eprintln!("Failed to send ROM to emulator thread: {}", e);
                                    }
                                    self.running.store(true, Relaxed);
                                }
                                Err(e) => {
                                    eprintln!("Failed to read ROM file {:?}: {}", path, e);
                                }
                            }
                        }
                    }
                });
                ui.separator();
                ui.heading("Emulator");

                // --- Start of Emulator Settings Grid ---
                let mut ips = self.target_ips.load(Relaxed);
                let mut fps = self.target_fps.load(Relaxed);

                egui::Grid::new("emulator_settings_grid")
                    .num_columns(2)
                    .spacing([40.0, 4.0]) // [column_spacing, row_spacing]
                    .show(ui, |ui| {
                        // Row 1: Target IPS
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            ui.label("Target IPS");
                        });
                        ui.add(egui::Slider::new(&mut ips, 1..=100_000));
                        ui.end_row();

                        // Row 2: Target FPS
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            ui.label("Target FPS");
                        });
                        const FPS_OPTIONS: &[u32] = &[30, 60, 120, 144, 240];
                        egui::ComboBox::new("fps_select", "")
                            .selected_text(format!("{} FPS", fps))
                            .show_ui(ui, |ui| {
                                for &fps_option in FPS_OPTIONS {
                                    ui.selectable_value(
                                        &mut fps,
                                        fps_option,
                                        format!("{} FPS", fps_option),
                                    );
                                }
                            });
                        ui.end_row();
                    });

                self.target_ips.store(ips, Relaxed);
                self.target_fps.store(fps, Relaxed);
                // --- End of Emulator Settings Grid ---

                ui.separator();
                ui.heading("Display");

                // --- Start of Display Settings Grid ---
                egui::Grid::new("display_settings_grid")
                    .num_columns(2)
                    .spacing([20.0, 4.0])
                    .show(ui, |ui| {
                        // Row 1: Game Scale
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            ui.label("Game Scale");
                        });
                        ui.add(egui::Slider::new(&mut self.game_scale, 1.0..=32.0));
                        ui.end_row();

                        // Row 2: On Pixel Color
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            ui.label("On Pixel: ");
                        });
                        ui.color_edit_button_srgba(&mut self.on_pixel_color);
                        ui.end_row();

                        // Row 3: Off Pixel Color
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            ui.label("Off Pixel:");
                        });
                        ui.color_edit_button_srgba(&mut self.off_pixel_color);
                        ui.end_row();
                    });
            });
    }

    fn render_info_panel(&self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Emulator Info")
            .default_open(true)
            .show(ui, |ui| {
                let frame_time = self.last_frame.elapsed();
                let fps = 1.0 / frame_time.as_secs_f64();
                let state = if self.running.load(Relaxed) {
                    "Running"
                } else {
                    "Stopped"
                };
                let instructions = self.instruction_counter.load(Relaxed);

                egui::Grid::new("info_grid").num_columns(2).show(ui, |ui| {
                    ui.label("GUI FPS:");
                    ui.label(format!("{:.1}", fps));
                    ui.end_row();

                    ui.label("Frame Time:");
                    ui.label(format!("{} ms", frame_time.as_millis()));
                    ui.end_row();

                    ui.label("Current State:");
                    ui.label(state);
                    ui.end_row();

                    ui.label("Instructions Executed:");
                    ui.label(format!("{instructions}"));
                    ui.end_row();

                    ui.label("Audio Status:");
                    ui.colored_label(
                        Color32::LIGHT_GREEN,
                        if self.sink.is_paused() { "OK" } else { "BEEP" },
                    );
                    ui.end_row();
                });
            });
    }

    fn render_game_screen(&mut self, ui: &mut egui::Ui) {
        let screen = self.last_state.get_screen();
        let image = render_screen(screen, self.on_pixel_color, self.off_pixel_color);
        self.display_texture
            .set(image, egui::TextureOptions::NEAREST);

        // Wrap the game in a frame
        Frame::dark_canvas(ui.style()).show(ui, |ui| {
            let image =
                egui::Image::new(&self.display_texture).fit_to_original_size(self.game_scale);
            ui.add(image);
        });
    }
}

struct DebuggerApp {
    state_rx: Receiver<CPU>,
    rom_tx: Sender<Vec<u8>>,
    last_state: CPU,
    keypad: Keypad,
    display_texture: egui::TextureHandle,
    _stream: OutputStream,
    sink: Sink,

    // Stats
    last_frame: Instant,
    instruction_counter: Arc<AtomicU64>,

    // Settings
    on_pixel_color: Color32,
    off_pixel_color: Color32,
    game_scale: f32,
    target_ips: Arc<AtomicU32>,
    target_fps: Arc<AtomicU32>,
    running: Arc<AtomicBool>,
}

impl eframe::App for DebuggerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.check_for_updates();
        self.keypad.0.store(Self::check_keyboard(ctx), Relaxed);

        SidePanel::left("left_panel")
            .resizable(true)
            .default_width(250.0)
            .show(ctx, |ui| {
                self.render_info_panel(ui);
                ui.separator();
                self.render_register_viewer(ui, &self.last_state);
                ui.separator();
                self.render_stack_viewer(ui, &self.last_state);
            });

        SidePanel::right("right_panel")
            .resizable(true)
            .default_width(200.0)
            .show(ctx, |ui| {
                self.render_settings_panel(ui);
            });
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Game Display Window");
            self.render_game_screen(ui);
            ui.separator();
            self.render_memory_viewer(ui, &self.last_state);
        });
        self.last_frame = Instant::now();
    }
}

fn main() -> eframe::Result {
    let rom = std::env::args()
        .nth(1)
        .map(|path| std::fs::read(path).expect("Failed to read ROM from path"));
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(egui::Vec2::new(1920.0, 1080.0))
            .with_min_inner_size(egui::Vec2::new(800.0, 600.0)),
        ..Default::default()
    };
    eframe::run_native(
        "Chip-8",
        native_options,
        Box::new(|cc| Ok(Box::new(DebuggerApp::new(cc, rom)))),
    )
}
