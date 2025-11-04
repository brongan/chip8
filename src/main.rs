use std::{
    sync::mpsc::Receiver,
    thread,
    time::{Duration, Instant},
};

use eframe::egui::{self, Vec2};
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

#[derive(Debug)]
struct Memory(pub [u8; 4096]);

impl Default for Memory {
    fn default() -> Self {
        Self([0; 4096])
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
struct Registers(pub [u8; 16]);

#[derive(Debug, strum::FromRepr, Copy, Clone)]
#[repr(u8)]
enum Register {
    V0,
    V1,
    V2,
    V3,
    V4,
    V5,
    V6,
    V7,
    V8,
    V9,
    VA,
    VB,
    VC,
    VD,
    VE,
    VF,
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
        let nib2 = (opcode >> 8) as u8 & 0x00FF;
        let register = Register::from_repr(nib2).unwrap();
        let addr = opcode & 0x0FFF;
        let x = &register;
        let y = Register::from_repr((opcode as u8) >> 4).unwrap();
        let n = opcode as u8 & 0xFF;
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
            _ => todo!(),
        }
    }
}

impl CPU {
    fn fetch(&self) -> u16 {
        let pc = self.pc as usize;
        (self.memory.0[pc] as u16) << 8 | self.memory.0[pc + 1] as u16
    }

    fn execute(&mut self, instruction: Instruction) {
        match instruction {
            Instruction::ClearScreen => self.screen = Screen::default(),
            Instruction::Jump(addr) => self.pc = addr,
            Instruction::SetRegister(register, val) => {
                self.registers.0[register as u8 as usize] = val
            }
            Instruction::Add(register, val) => self.registers.0[register as u8 as usize] += val,
            Instruction::SetIndex(val) => self.index = val,
            Instruction::Display(x, y, val) => todo!(),
        }
    }

    pub fn tick(&mut self) {
        let instruction = self.fetch();
        let instruction = Instruction::decode(instruction);
        self.execute(instruction);
        self.pc += 2;
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
    fn new() -> Self {
        Self {
            cpu: CPU::default(),
            frame_counter: FrameCounter::default(),
        }
    }

    fn tick(&mut self) -> Option<Screen> {
        self.cpu.tick();
        if self.frame_counter.tick() {
            self.cpu.tick_timers();
            return Some(self.cpu.screen());
        }
        None
    }
}

struct DebuggerApp {
    rx: Receiver<Screen>,
    screen: Screen,
    display_texture: egui::TextureHandle,
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
    fn new(cc: &eframe::CreationContext) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        let ctx = cc.egui_ctx.clone();
        let mut emulator = Emulator::new();
        thread::spawn(move || {
            loop {
                let start = Instant::now();
                let ips = 700;
                let instruction_time = Duration::from_secs_f64(1.0 / ips as f64);
                if let Some(frame) = emulator.tick() {
                    if tx.send(frame).is_err() {
                        break;
                    }
                }
                sleep(instruction_time - (Instant::now().duration_since(start)));
                ctx.request_repaint();
            }
        });

        let image = egui::ColorImage::new([64, 32], vec![egui::Color32::BLACK; 64 * 32]);
        let display_texture = cc
            .egui_ctx
            .load_texture("LCD", image, egui::TextureOptions::NEAREST);

        Self {
            rx,
            screen: Screen::default(),
            display_texture,
        }
    }

    fn check_for_updates(&mut self) {
        if let Ok(frame) = self.rx.try_recv() {
            self.screen = frame;
        }
    }
}

impl eframe::App for DebuggerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.check_for_updates();
        egui::CentralPanel::default().show(ctx, |ui| {
            let image = render_screen(&self.screen);
            self.display_texture
                .set(image, egui::TextureOptions::NEAREST);
            ui.add(egui::Image::new(&self.display_texture).fit_to_original_size(4.0));
        });
    }
}

fn main() -> eframe::Result {
    let rom = std::env::args().nth(1).unwrap();
    eframe::run_native(
        "Chips8",
        eframe::NativeOptions::default(),
        Box::new(|cc| Ok(Box::new(DebuggerApp::new(cc)))),
    )
}
