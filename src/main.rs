#![feature(iter_array_chunks)]
use eframe::egui::{
    self, Align, Button, CentralPanel, Color32, Frame, Key, Layout, RichText, ScrollArea,
    SidePanel, TextStyle, TopBottomPanel, Vec2,
};
use rodio::source::SineWave;
use rodio::{OutputStream, OutputStreamBuilder, Sink, Source};
use std::sync::mpsc::{Receiver, Sender};
use std::thread;
use std::time::Instant;
use strum::IntoEnumIterator;

use chip8::{CPU, Instruction, Quirks, Register, Registers, Screen};

const ACTIVE_COLOR: Color32 = Color32::from_rgb(50, 100, 200);

struct Rom {
    path: String,
    contents: Vec<u8>,
}

impl Rom {
    pub fn new(path: String) -> Self {
        let contents = std::fs::read(&path).expect("Failed to read ROM from path");
        Self { path, contents }
    }
}

struct DebuggerApp {
    cpu: CPU,
    display_texture: egui::TextureHandle,
    _stream: OutputStream,
    sink: Sink,
    rom: Option<Rom>,
    rom_tx: Sender<(Vec<u8>, String)>,
    rom_rx: Receiver<(Vec<u8>, String)>,

    // Stats
    last_frame_time: Instant,
    instruction_counter: u64,

    // Timing
    accumulator: f32,
    timer_accumulator: f32,

    // Settings
    on_pixel_color: Color32,
    off_pixel_color: Color32,
    game_scale: f32,
    target_ips: u32,
    running: bool,
    quirks: Quirks,
}

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

fn handle_rom_dialog(rom_tx: Sender<(Vec<u8>, String)>) {
    if let Some(path) = rfd::FileDialog::new()
        .add_filter("CHIP-8 ROM", &["ch8", "rom"])
        .add_filter("All Files", &["*"])
        .pick_file()
    {
        if let Ok(rom_data) = std::fs::read(&path) {
            let path_str = path.to_string_lossy().to_string();
            if rom_tx.send((rom_data, path_str)).is_err() {
                eprintln!("Failed to send new ROM to main thread.");
            }
        }
    }
}

impl DebuggerApp {
    fn new(cc: &eframe::CreationContext, rom_path: Option<String>) -> Self {
        let ctx = cc.egui_ctx.clone();
        ctx.set_visuals(egui::Visuals::dark());

        let rom = rom_path.map(Rom::new);
        let running = rom.is_some();
        let (rom_tx, rom_rx) = std::sync::mpsc::channel();
        let cpu = CPU::new(rom.as_ref().map(|rom| &rom.contents));

        let image = egui::ColorImage::new([64, 32], vec![Color32::BLACK; 64 * 32]);
        let display_texture = cc
            .egui_ctx
            .load_texture("LCD", image, egui::TextureOptions::NEAREST);

        let stream = OutputStreamBuilder::open_default_stream().expect("open default audio stream");
        let sink = Sink::connect_new(&stream.mixer());
        let beep_sound = SineWave::new(440.0).amplify(0.20);
        sink.append(beep_sound);
        sink.pause();

        Self {
            cpu,
            display_texture,
            _stream: stream,
            sink,
            rom,
            rom_tx,
            rom_rx,
            on_pixel_color: Color32::WHITE,
            off_pixel_color: Color32::BLACK,
            game_scale: 8.0,
            target_ips: 700,
            running,
            last_frame_time: Instant::now(),
            instruction_counter: 0,
            accumulator: 0.0,
            timer_accumulator: 0.0,
            quirks: Quirks::default(),
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

    fn render_cpu_state(ui: &mut egui::Ui, cpu: &CPU) {
        egui::CollapsingHeader::new("Chip-8 CPU State")
            .default_open(true)
            .show_unindented(ui, |ui| {
                ui.style_mut().override_text_style = Some(TextStyle::Monospace);

                Self::render_registers(
                    ui,
                    cpu.get_pc(),
                    cpu.fetch(),
                    cpu.get_index(),
                    cpu.get_delay_timer(),
                    cpu.get_sound_timer(),
                    cpu.get_registers(),
                );
                ui.separator();

                Self::render_stack(ui, &cpu.get_stack());
            });
    }

    fn render_registers(
        ui: &mut egui::Ui,
        pc: u16,
        ir: u16,
        index: u16,
        delay: u8,
        sound: u8,
        registers: &Registers,
    ) {
        egui::Grid::new("registers").show(ui, |ui| {
            ui.style_mut().override_text_style = Some(TextStyle::Monospace);
            ui.label(format!("PC\n0x{pc:04X}\n{pc}"));
            ui.label(format!("IR\n0x{ir:04X}\n{ir}"));
            ui.label(format!("I\n0x{index:04X}\n{index}"));
            ui.end_row();
            ui.label(format!("DT (delay)\n0x{delay:04X}\n{delay}"));
            ui.label(format!("ST (sound)\n0x{sound:04X}\n{sound}"));
            ui.end_row();
            for row in Register::iter().array_chunks::<4>() {
                for reg in row {
                    let val = registers.get(reg);
                    ui.label(format!("{reg}\n0x{val:02X}\n{val}"));
                }
                ui.end_row();
            }
        });
    }

    fn render_stack(ui: &mut egui::Ui, stack: &[u16]) {
        let sp = stack.len();
        ui.label(format!("SP: 0x{sp:04X} {sp}"));
        egui::CollapsingHeader::new("Stack Viewer")
            .default_open(true)
            .show_unindented(ui, |ui| {
                ScrollArea::vertical().max_height(150.0).show(ui, |ui| {
                    egui::Grid::new("stack_grid")
                        .num_columns(2)
                        .striped(true)
                        .show(ui, |ui| {
                            ui.label(RichText::new("Depth").strong());
                            ui.label(RichText::new("Contents").strong());
                            ui.end_row();

                            for (i, &addr) in stack.iter().enumerate().rev() {
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

    fn render_memory(ui: &mut egui::Ui, memory: &[u8], pc: u16) {
        egui::CollapsingHeader::new("Memory Viewer")
            .default_open(true)
            .show_unindented(ui, |ui| {
                ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        egui::Grid::new("memory_grid")
                            .spacing([15.0, 1.0])
                            .striped(true)
                            .show(ui, |ui| {
                                ui.style_mut().override_text_style = Some(TextStyle::Monospace);
                                ui.spacing_mut().item_spacing.x = 2.0;

                                for (i, chunk) in memory.chunks(16).enumerate() {
                                    let addr = i * 16;
                                    let color = if (addr..addr + 16).contains(&(pc as usize)) {
                                        Color32::YELLOW
                                    } else {
                                        Color32::LIGHT_GREEN
                                    };
                                    let binary = chunk
                                        .into_iter()
                                        .map(|byte| format!("{:02X}", byte))
                                        .collect::<String>();

                                    let ascii_art = chunk
                                        .iter()
                                        .map(|&b| match b {
                                            b' '..=b'~' => b as char,
                                            _ => '.',
                                        })
                                        .collect::<String>();

                                    ui.colored_label(color, format!("0x{:04X}", addr));
                                    ui.colored_label(color, binary);
                                    ui.colored_label(color, ascii_art);
                                    ui.end_row();
                                }
                            });
                    });
            });
    }

    fn render_settings_panel(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Settings Menu")
            .default_open(true)
            .show_unindented(ui, |ui| {
                ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);

                let button_size = egui::vec2(250.0, 30.0);
                let pause_run = Button::new(if self.running { "Pause" } else { "Run" })
                    .min_size(button_size)
                    .fill(ACTIVE_COLOR);
                let load_rom = Button::new("Load ROM")
                    .min_size(button_size)
                    .fill(ACTIVE_COLOR);
                let reset = Button::new("Reset")
                    .min_size(button_size)
                    .fill(ACTIVE_COLOR);
                let reload_rom = Button::new("Reload ROM")
                    .min_size(button_size)
                    .fill(ACTIVE_COLOR);
                let step = Button::new("Step").min_size(button_size).fill(ACTIVE_COLOR);
                let ten = Button::new("Step 10")
                    .min_size(button_size)
                    .fill(ACTIVE_COLOR);

                ui.heading("Controls");
                ui.horizontal_centered(|ui| {
                    if ui.add(pause_run).clicked() {
                        self.running = !self.running;
                    }
                });
                ui.horizontal_centered(|ui| {
                    if ui.add(reset).clicked() {
                        self.instruction_counter = 0;
                        self.cpu = CPU::new(None);
                    }
                });
                ui.horizontal_centered(|ui| {
                    if ui.add(load_rom).clicked() {
                        self.instruction_counter = 0;
                        let rom_tx = self.rom_tx.clone();
                        thread::spawn(move || {
                            handle_rom_dialog(rom_tx);
                        });
                    }
                });
                ui.horizontal_centered(|ui| {
                    if ui.add(reload_rom).clicked() {
                        self.instruction_counter = 0;
                        self.cpu = CPU::new(self.rom.as_ref().map(|rom| &rom.contents));
                    }
                });

                ui.heading("Debug");

                ui.horizontal_centered(|ui| {
                    if ui.add(step).clicked() {
                        self.cpu.tick(&self.quirks);
                        self.instruction_counter += 1;
                    }
                });
                ui.horizontal_centered(|ui| {
                    if ui.add(ten).clicked() {
                        for _ in 0..10 {
                            self.cpu.tick(&self.quirks);
                            self.instruction_counter += 1;
                        }
                    }
                });

                ui.heading("Emulator");

                egui::Grid::new("emulator_settings_grid")
                    .num_columns(2)
                    .spacing([40.0, 4.0]) // [column_spacing, row_spacing]
                    .show(ui, |ui| {
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            ui.label("Target IPS");
                        });
                        ui.add(egui::Slider::new(&mut self.target_ips, 1..=10_000));
                        ui.end_row();
                    });
                ui.checkbox(&mut self.quirks.vf_reset, "VF Reset");
                ui.checkbox(&mut self.quirks.memory_increment, "Memory Increment");
                ui.checkbox(&mut self.quirks.clipping, "Clipping");
                ui.checkbox(&mut self.quirks.display_wait, "Display Wait");
                ui.checkbox(&mut self.quirks.shift_vy, "Shifting");
                ui.checkbox(&mut self.quirks.jumping, "Jumping");

                ui.heading("Display");

                egui::Grid::new("display_settings_grid")
                    .num_columns(2)
                    .spacing([20.0, 4.0])
                    .show(ui, |ui| {
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            ui.label("Game Scale");
                        });
                        ui.add(egui::Slider::new(&mut self.game_scale, 1.0..=32.0));
                        ui.end_row();

                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            ui.label("On Pixel: ");
                        });
                        ui.color_edit_button_srgba(&mut self.on_pixel_color);
                        ui.end_row();

                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            ui.label("Off Pixel:");
                        });
                        ui.color_edit_button_srgba(&mut self.off_pixel_color);
                        ui.end_row();
                    });
            });
    }

    fn render_info_panel(&self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Chip-8 Emulator Info")
            .default_open(true)
            .show_unindented(ui, |ui| {
                let frame_time = self.last_frame_time.elapsed();
                let fps = 1.0 / frame_time.as_secs_f64();
                let state = if self.running { "Running" } else { "Stopped" };

                let rom = self.rom.as_ref().map_or("None", |rom| &rom.path);
                ui.label(format!("ROM: {rom}"));

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
                    ui.label(format!("{}", self.instruction_counter));
                    ui.end_row();

                    ui.label("Audio Status:");
                    let label = if self.sink.is_paused() { "OK" } else { "BEEP" };
                    ui.colored_label(Color32::LIGHT_GREEN, label);
                    ui.end_row();
                });
            });
    }

    fn render_game_screen(&mut self, ui: &mut egui::Ui) {
        let screen = self.cpu.get_screen();
        let image = render_screen(screen, self.on_pixel_color, self.off_pixel_color);
        self.display_texture
            .set(image, egui::TextureOptions::NEAREST);
        Frame::dark_canvas(ui.style()).show(ui, |ui| {
            let image =
                egui::Image::new(&self.display_texture).fit_to_original_size(self.game_scale);
            ui.add(image);
        });
    }

    fn render_disassembler(ui: &mut egui::Ui, cpu: &CPU) {
        egui::CollapsingHeader::new("Disassembly")
            .default_open(true)
            .show_unindented(ui, |ui| {
                ScrollArea::vertical().show(ui, |ui| {
                    egui::Grid::new("disassembler").show(ui, |ui| {
                        let instruction_stream = ((cpu.get_pc() as usize)..cpu.get_memory().len())
                            .step_by(2)
                            .filter_map(|addr| {
                                let instruction = (cpu.get_memory()[addr] as u16) << 8
                                    | cpu.get_memory()[addr + 1] as u16;
                                Instruction::decode(instruction)
                                    .map(|inst| (addr, instruction, inst))
                            });

                        for (addr, instruction, decoded) in instruction_stream {
                            ui.label(format!("0x{addr:04x}"));
                            ui.label(format!("{instruction:04x}"));
                            ui.label(format!("{decoded}"));
                            ui.end_row();
                        }
                    });
                });
            });
    }

    fn show_keyboard(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        self.cpu
            .get_keypad_mut()
            .set_state(Self::check_keyboard(ctx));
        egui::CollapsingHeader::new("Keypad")
            .default_open(true)
            .show_unindented(ui, |ui| {
                egui::Grid::new("keypad")
                    .spacing(Vec2::new(5.0, 5.0))
                    .show(ui, |ui| {
                        let key_layout: [u8; 16] = [
                            0x1, 0x2, 0x3, 0xC, 0x4, 0x5, 0x6, 0xD, 0x7, 0x8, 0x9, 0xE, 0xA, 0x0,
                            0xB, 0xF,
                        ];
                        for row in key_layout.iter().array_chunks::<4>() {
                            for &key_index in row {
                                let fill_color = if self.cpu.get_keypad_mut().is_pressed(key_index)
                                {
                                    ACTIVE_COLOR
                                } else {
                                    ui.visuals().widgets.inactive.bg_fill
                                };

                                let btn = Button::new(format!("{:X}", key_index))
                                    .min_size(Vec2::new(60.0, 60.0))
                                    .fill(fill_color);

                                if ui.add(btn).clicked() {
                                    self.cpu.get_keypad_mut().enable_key(key_index);
                                };
                            }
                            ui.end_row();
                        }
                    });
            });
    }
}

impl eframe::App for DebuggerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Ok((contents, path)) = self.rom_rx.try_recv() {
            self.cpu = CPU::new(Some(&contents));
            self.rom = Some(Rom { path, contents });
            self.running = true;
            self.instruction_counter = 0;
        }

        let dt = self.last_frame_time.elapsed().as_secs_f32();
        self.last_frame_time = Instant::now();

        if self.running {
            self.accumulator += dt;
            let cycle_duration = 1.0 / self.target_ips as f32;
            while self.accumulator >= cycle_duration {
                self.cpu.tick(&self.quirks);
                self.instruction_counter += 1;
                self.accumulator -= cycle_duration;
            }

            self.timer_accumulator += dt;
            let timer_step = 1.0 / 60.0;
            while self.timer_accumulator >= timer_step {
                self.cpu.tick_timers();
                self.timer_accumulator -= timer_step;
            }
        }

        if self.cpu.is_beep() {
            self.sink.play();
        } else {
            self.sink.pause();
        }

        SidePanel::left("left_panel")
            .resizable(true)
            .default_width(250.0)
            .show(ctx, |ui| {
                self.render_info_panel(ui);
                ui.separator();
                Self::render_cpu_state(ui, &self.cpu);
            });

        SidePanel::right("right_panel")
            .resizable(true)
            .show(ctx, |ui| Self::render_disassembler(ui, &self.cpu));

        TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    self.render_settings_panel(ui);
                });
                ui.vertical(|ui| {
                    self.render_game_screen(ui);
                });
            });
        });

        CentralPanel::default().show(ctx, |ui| {
            let available_height = ui.available_height();
            ui.horizontal(|ui| {
                ui.set_min_height(available_height);
                ui.vertical(|ui| {
                    self.show_keyboard(ctx, ui);
                });
                ui.vertical(|ui| {
                    Self::render_memory(ui, self.cpu.get_memory(), self.cpu.get_pc());
                });
            });
        });

        ctx.request_repaint();
    }
}

fn main() -> eframe::Result {
    let rom = std::env::args().nth(1);
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
