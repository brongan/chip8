use std::{
    sync::mpsc::Receiver,
    thread,
    time::{Duration, Instant},
};

use eframe::egui::{self, Vec2};
use rodio::{OutputStream, Sink, Source};
use spin_sleep::sleep;

#[derive(Default, Debug)]
struct CPU {
    pc: u16,
    index: u16,
    sp: u16,
    delay_timer: Timer,
    sound_timer: Timer,
    registers: Registers,
    memory: Memory,
    screen: Screen,
}

impl CPU {
    fn new(rom: Vec<u8>) -> Self {
        let font = [
            0xF0, 0x90, 0x90, 0x90, 0xF0, // 0
            0x20, 0x60, 0x20, 0x20, 0x70, // 1
            0xF0, 0x10, 0xF0, 0x80, 0xF0, // 2
            0xF0, 0x10, 0xF0, 0x10, 0xF0, // 3
            0x90, 0x90, 0xF0, 0x10, 0x10, // 4
            0xF0, 0x80, 0xF0, 0x10, 0xF0, // 5
            0xF0, 0x80, 0xF0, 0x90, 0xF0, // 6
            0xF0, 0x10, 0x20, 0x40, 0x40, // 7
            0xF0, 0x90, 0xF0, 0x90, 0xF0, // 8
            0xF0, 0x90, 0xF0, 0x10, 0xF0, // 9
            0xF0, 0x90, 0xF0, 0x90, 0x90, // A
            0xE0, 0x90, 0xE0, 0x90, 0xE0, // B
            0xF0, 0x80, 0x80, 0x80, 0xF0, // C
            0xE0, 0x90, 0x90, 0x90, 0xE0, // D
            0xF0, 0x80, 0xF0, 0x80, 0xF0, // E
            0xF0, 0x80, 0xF0, 0x80, 0x80, // F;
        ];
        let mut memory = Memory::default();
        let len = std::cmp::min(rom.len(), memory.0.len() - 0x200);
        memory.0[0x50..=0x9f].copy_from_slice(&font);
        memory.0[0x200..len + 0x200].copy_from_slice(&rom[0..len]);
        Self {
            memory,
            pc: 0x200,
            ..Default::default()
        }
    }
}

#[derive(Debug)]
struct Memory([u8; 4096]);

impl Default for Memory {
    fn default() -> Self {
        Self([0; 4096])
    }
}

impl Memory {
    fn get(&self, addr: u16) -> u8 {
        self.0[addr as usize]
    }

    fn set(&mut self, addr: u16, val: u8) {
        self.0[addr as usize] = val
    }

    fn get_mut(&mut self, addr: u16) -> &mut u8 {
        &mut self.0[addr as usize]
    }
}

#[derive(Debug, Clone)]
struct Screen([[bool; 64]; 32]);

impl Default for Screen {
    fn default() -> Self {
        Self([[false; 64]; 32])
    }
}

#[derive(Default, Debug)]
struct Registers([u8; 16]);

impl Registers {
    fn get(&self, register: Register) -> u8 {
        self.0[register as u8 as usize]
    }

    fn set(&mut self, register: Register, val: u8) {
        self.0[register as u8 as usize] = val
    }

    fn get_mut(&mut self, register: Register) -> &mut u8 {
        &mut self.0[register as u8 as usize]
    }
}

#[derive(Debug, strum::FromRepr, Copy, Clone)]
#[repr(u8)]
enum Register {
    V0 = 0x0,
    V1 = 0x1,
    V2 = 0x2,
    V3 = 0x3,
    V4 = 0x4,
    V5 = 0x5,
    V6 = 0x6,
    V7 = 0x7,
    V8 = 0x8,
    V9 = 0x9,
    VA = 0xa,
    VB = 0xb,
    VC = 0xc,
    VD = 0xd,
    VE = 0xe,
    VF = 0xf,
}

#[derive(Default, Debug)]
struct Timer(u8);

impl Timer {
    /// Returns true if it reached 0.
    fn tick(&mut self) -> bool {
        let ret = self.0 == 1;
        self.0 = self.0.saturating_sub(1);
        ret
    }
}

#[derive(Debug)]
enum Instruction {
    ClearScreen,
    Jump(u16),
    SetRegister(Register, u8),
    Add(Register, u8),
    SetIndex(u16),
    Display(Register, Register, u8),
}

impl Instruction {
    fn decode(opcode: u16) -> Self {
        use Instruction::*;
        match opcode {
            0x00E0 => return ClearScreen,
            _ => (),
        }
        let nib1 = (opcode >> 12) as u8;
        let nib2 = (opcode >> 8) as u8 & 0xF;
        let register = Register::from_repr(nib2).unwrap();
        let addr = opcode & 0x0FFF;
        let x = &register;
        let y = Register::from_repr((opcode as u8) >> 4).unwrap();
        let n = opcode as u8 & 0xF;
        match nib1 {
            0x1 => Jump(opcode & 0x0FFF),
            // 6XNN (set register VX)
            0x6 => SetRegister(register, opcode as u8),
            // 7XNN (add value to register VX)
            0x7 => Add(register, opcode as u8),
            // ANNN (set index register I)
            0xA => SetIndex(addr),
            // DXYN (display/draw)
            0xD => Display(*x, y, n),
            b => todo!("Unknown opcode: 0x{:04x}", b),
        }
    }
}

impl CPU {
    fn fetch(&self) -> u16 {
        let pc = self.pc as usize;
        (self.memory.0[pc] as u16) << 8 | self.memory.0[pc + 1] as u16
    }

    fn execute(&mut self, instruction: Instruction) -> u16 {
        match instruction {
            Instruction::ClearScreen => {
                self.screen = Screen::default();
            }
            Instruction::Jump(addr) => return addr,
            Instruction::SetRegister(register, val) => {
                self.registers.set(register, val);
            }
            Instruction::Add(register, val) => {
                *self.registers.get_mut(register) += val;
            }
            Instruction::SetIndex(val) => {
                self.index = val;
            }
            Instruction::Display(x, y, height) => {
                let vx = self.registers.get(x);
                let vy = self.registers.get(y);
                let x = vx % 64;
                let y = vy % 32;
                let vf = self.registers.get_mut(Register::VF);
                *vf = 0;
                for (j, y) in (y..std::cmp::min(32, y + height)).enumerate() {
                    let row = self.memory.get(self.index + j as u16);
                    for (i, x) in (x..std::cmp::min(64, x + 8)).enumerate() {
                        if row & (0b1 << (7 - i)) > 0 {
                            if self.screen.0[y as usize][x as usize] {
                                self.screen.0[y as usize][x as usize] = false;
                                *vf = 1;
                            } else {
                                self.screen.0[y as usize][x as usize] = true;
                            }
                        }
                    }
                }
            }
        }
        self.pc + 2
    }

    pub fn tick(&mut self) {
        let instruction = self.fetch();
        let instruction = Instruction::decode(instruction);
        self.pc = self.execute(instruction);
    }

    /// The caller should tick the timers at a 60hz frequency.
    /// Returns true if it beeps.
    pub fn tick_timers(&mut self) -> bool {
        self.delay_timer.tick();
        self.sound_timer.tick()
    }

    /// The caller should retrieve the screen at a 60hz frequency.
    pub fn screen(&self) -> Screen {
        self.screen.clone()
    }
}

#[derive(Default, Debug)]
struct FrameCounter {
    val: usize,
    remainder: u8,
}

/// Renders a frame every 11.66666 ticks
impl FrameCounter {
    /// Returns true if a frame should be rendered.
    fn tick(&mut self) -> bool {
        self.val += 1;
        let limit = if self.remainder == 2 { 11 } else { 12 };
        if self.val >= limit {
            self.val = 0;
            self.remainder = (self.remainder + 1) % 3;
            true
        } else {
            false
        }
    }
}

#[derive(Debug)]
struct Emulator {
    cpu: CPU,
    frame_counter: FrameCounter,
}

impl Emulator {
    fn new(rom: Vec<u8>) -> Self {
        Self {
            cpu: CPU::new(rom),
            frame_counter: FrameCounter::default(),
        }
    }

    fn tick(&mut self) -> Option<EmulatorState> {
        self.cpu.tick();
        if self.frame_counter.tick() {
            return Some(EmulatorState {
                beep: self.cpu.tick_timers(),
                screen: self.cpu.screen(),
            });
        }
        None
    }
}

struct EmulatorState {
    beep: bool,
    screen: Screen,
}

struct DebuggerApp {
    rx: Receiver<EmulatorState>,
    screen: Screen,
    display_texture: egui::TextureHandle,
    stream: OutputStream,
    sink: Sink,
}

fn render_screen(screen: &Screen) -> egui::ColorImage {
    let pixels: Vec<egui::Color32> = screen
        .0
        .iter()
        .flat_map(|row| row.iter())
        .map(|pixel| egui::Color32::from_gray(*pixel as u8 * 255))
        .collect();

    egui::ColorImage {
        size: [64, 32],
        pixels,
        source_size: Vec2::default(),
    }
}

impl DebuggerApp {
    fn new(cc: &eframe::CreationContext, rom: Vec<u8>) -> Self {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let ctx = cc.egui_ctx.clone();
        let mut emulator = Emulator::new(rom);
        thread::spawn(move || {
            loop {
                let start = Instant::now();
                let ips = 700;
                let instruction_time = Duration::from_secs_f64(1.0 / ips as f64);
                if let Some(state) = emulator.tick() {
                    if tx.send(state).is_err() {
                        break;
                    }
                    ctx.request_repaint();
                }
                sleep(instruction_time - (Instant::now().duration_since(start)));
            }
        });

        let image = egui::ColorImage::new([64, 32], vec![egui::Color32::BLACK; 64 * 32]);
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
            rx,
            screen: Screen::default(),
            display_texture,
            stream,
            sink,
        }
    }

    fn check_for_updates(&mut self) {
        if let Ok(state) = self.rx.try_recv() {
            self.screen = state.screen;
            if state.beep {
                self.sink.play();
            } else {
                self.sink.pause();
            }
        }
    }

    fn render_content(&mut self, ui: &mut egui::Ui) {
        self.check_for_updates();
        let image = render_screen(&self.screen);
        self.display_texture
            .set(image, egui::TextureOptions::NEAREST);

        let image = render_screen(&self.screen);
        self.display_texture
            .set(image, egui::TextureOptions::NEAREST);
        ui.add(egui::Image::new(&self.display_texture).fit_to_original_size(16.0));
    }
}

impl eframe::App for DebuggerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            self.render_content(ui);
        });
    }
}

fn main() -> eframe::Result {
    let rom = std::env::args().nth(1).unwrap();
    let rom = std::fs::read(rom).unwrap();
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size(egui::Vec2::new(1024.0, 512.0)),
        ..Default::default()
    };
    eframe::run_native(
        "Chip-8",
        native_options,
        Box::new(|cc| Ok(Box::new(DebuggerApp::new(cc, rom)))),
    )
}
