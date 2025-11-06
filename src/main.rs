use std::{
    sync::mpsc::{Receiver, SyncSender},
    thread,
    time::{Duration, Instant},
};

use eframe::egui::{self, Key, Vec2};
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
    keypad: Keypad,
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

    fn is_beep(&self) -> bool {
        self.sound_timer.get() > 0
    }

    fn set_keypad(&mut self, keypad: Keypad) {
        self.keypad = keypad;
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

#[derive(Debug, strum::FromRepr, Copy, Clone, strum::EnumIter)]
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
    fn tick(&mut self) {
        self.0 = self.0.saturating_sub(1);
    }

    fn get(&self) -> u8 {
        self.0
    }

    fn set(&mut self, val: u8) {
        self.0 = val
    }
}

#[derive(Debug)]
enum Cond {
    /// VX equals NN
    Eq(Register, u8),
    /// VX does not equal NN
    Neq(Register, u8),
    /// VX equals VY
    EqReg(Register, Register),
    /// VX does not equal VY
    NeqReg(Register, Register),
}

#[derive(Debug)]
enum Instruction {
    /// Adds NN to VX (carry flag is not changed).
    Add(Register, u8),
    /// Adds VX to I. VF is not affected.
    AddIndex(Register),
    /// Adds VY to VX. VF is set to 1 when there's an overflow, and to 0 when there is not.
    AddReg(Register, Register),
    /// Sets VX to VX and VY. (bitwise AND operation).
    And(Register, Register),
    /// Sets VX to the value of VY.
    Assign(Register, Register),
    /// Stores the binary-coded decimal representation of VX,
    /// with the hundreds digit in memory at location in I,
    /// the tens digit at location I+1, and the ones digit at location I+2
    BinaryDecimalConversion(Register),
    /// Calls machine code routine at address NNN.
    Call(u16),
    /// Calls subroutine at NNN.
    CallSubroutine(u16),
    /// Clears the screen.
    DisplayClear,
    /// Skips the next instruction if Cond
    CondSkip(Cond),
    /// Draws a sprite at coordinate (VX, VY)
    Display(Register, Register, u8),
    /// Sets I to the location of the sprite for the character in VX(only consider the lowest nibble).
    /// Characters 0-F (in hexadecimal) are represented by a 4x5 font.
    FontCharacter(Register),
    /// Sets VX to the value of the delay timer.
    GetDelay(Register),
    /// A key press is awaited, and then stored in VX
    /// (blocking operation, all instruction halted until next key event, delay and sound timers should continue processing)
    GetKey(Register),
    /// Jumps to address NNN.
    Jump(u16),
    /// Jumps to the address NNN plus V0.
    JumpOffset(u16),
    /// Fills from V0 to VX (including VX) with values from memory, starting at address I. The offset from I is increased by 1 for each value read, but I itself is left unmodified.
    LoadMemory(Register),
    /// Sets VX to VX or VY. (bitwise OR operation).
    Or(Register, Register),
    /// Sets VX to the result of a bitwise and
    /// operation on a random number (Typically: 0 to 255) and NN.
    Rand(Register, u8),
    /// Returns from a subroutine.
    Return,
    /// Sets the delay timer to VX.
    SetDelay(Register),
    /// Sets I to the address NNN.
    SetIndex(u16),
    /// Sets VX to NN
    SetRegister(Register, u8),
    /// Sets the sound timer to VX.
    SetSound(Register),
    /// Sets VX to VY minus VX.
    /// VF is set to 0 when there's an underflow,
    /// and 1 when there is not.
    /// (i.e. VF set to 1 if VY >= VX).
    /// Shifts VX to the left by 1, then sets VF to 1 if the most significant bit of VX prior to that shift was set, or to 0 if it was unset.
    ShiftLeft(Register, Register),
    /// Shifts VX to the right by 1,
    /// then stores the least significant bit of VX prior to the shift into VF
    ShiftRight(Register, Register),
    /// Skips the next instruction if the key stored in VX(only consider the lowest nibble) is pressed
    SkipIfKey(Register),
    /// Skips the next instruction if the key stored in VX(only consider the lowest nibble) is not pressed
    SkipIfNotKey(Register),
    /// Stores from V0 to VX (including VX) in memory, starting at address I. The offset from I is increased by 1 for each value written, but I itself is left unmodified
    StoreMemory(Register),
    /// Complex
    Subtract(Register, Register),
    /// Sets VX to VX xor VY
    Xor(Register, Register),
}

impl Instruction {
    fn decode(opcode: u16) -> Self {
        use Instruction::*;
        let nib1 = (opcode >> 12) as u8;
        let nib2 = (opcode >> 8) as u8 & 0xF;
        let addr = opcode & 0x0FFF;
        let x = Register::from_repr(nib2).unwrap();
        let y = Register::from_repr((opcode as u8) >> 4).unwrap();
        let nn = opcode as u8;
        let n = nn & 0xF;
        match nib1 {
            0x0 if opcode == 0x00E0 => DisplayClear,
            0x0 if opcode == 0x00EE => Return,
            0x0 => Call(addr),
            0x1 => Jump(opcode & 0x0FFF),
            0x2 => CallSubroutine(addr),
            0x3 => CondSkip(Cond::Eq(x, nn)),
            0x4 => CondSkip(Cond::Neq(x, nn)),
            0x5 if n == 0x0 => CondSkip(Cond::EqReg(x, y)),
            0x6 => SetRegister(x, opcode as u8),
            0x7 => Add(x, opcode as u8),
            0x8 if n == 0x0 => Assign(x, y),
            0x8 if n == 0x1 => Or(x, y),
            0x8 if n == 0x2 => And(x, y),
            0x8 if n == 0x3 => Xor(x, y),
            0x8 if n == 0x4 => AddReg(x, y),
            0x8 if n == 0x5 => Subtract(x, y),
            0x8 if n == 0x6 => ShiftRight(x, y),
            0x8 if n == 0x7 => Subtract(y, x),
            0x8 if n == 0xe => ShiftLeft(x, y),
            0x9 if n == 0x0 => CondSkip(Cond::NeqReg(x, y)),
            0xA => SetIndex(addr),
            0xB => JumpOffset(addr),
            0xC => Rand(x, nn),
            0xD => Display(x, y, n),
            0xE if nn == 0x9E => SkipIfKey(x),
            0xE if nn == 0xA1 => SkipIfNotKey(x),
            0xF if nn == 0x07 => GetDelay(x),
            0xF if nn == 0x0A => GetKey(x),
            0xF if nn == 0x15 => SetDelay(x),
            0xF if nn == 0x18 => SetSound(x),
            0xF if nn == 0x1E => AddIndex(x),
            0xF if nn == 0x29 => FontCharacter(x),
            0xF if nn == 0x33 => BinaryDecimalConversion(x),
            0xF if nn == 0x55 => StoreMemory(x),
            0xF if nn == 0x65 => LoadMemory(x),
            _ => todo!("Unknown opcode: 0x{:04x}", opcode),
        }
    }
}

impl CPU {
    fn fetch(&self) -> u16 {
        let pc = self.pc as usize;
        (self.memory.0[pc] as u16) << 8 | self.memory.0[pc + 1] as u16
    }

    fn push(&mut self, addr: u16) {
        self.memory.set(self.sp, (addr >> 8) as u8);
        self.memory.set(self.sp + 1, addr as u8);
        self.sp += 2;
    }

    fn pop(&mut self) -> u16 {
        self.sp -= 2;
        let hi = self.memory.get(self.sp);
        let lo = self.memory.get(self.sp + 1);
        (hi as u16) << 8 | lo as u16
    }

    fn execute(&mut self, instruction: Instruction) -> u16 {
        match instruction {
            Instruction::DisplayClear => self.screen = Screen::default(),
            Instruction::CallSubroutine(addr) => {
                self.push(self.pc);
                return addr;
            }
            Instruction::Return => return self.pop(),
            Instruction::Jump(addr) => return addr,
            Instruction::SetRegister(register, val) => self.registers.set(register, val),
            Instruction::Add(register, val) => *self.registers.get_mut(register) += val,
            Instruction::SetIndex(val) => self.index = val,
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
            Instruction::AddIndex(register) => todo!(),
            Instruction::AddReg(register, register1) => todo!(),
            Instruction::And(register, register1) => todo!(),
            Instruction::BinaryDecimalConversion(register) => todo!(),
            Instruction::CondSkip(cond) => todo!(),
            Instruction::FontCharacter(register) => todo!(),
            Instruction::GetDelay(register) => todo!(),
            Instruction::GetKey(register) => todo!(),
            Instruction::JumpOffset(_) => todo!(),
            Instruction::LoadMemory(x) => {
                let x = x as u8;
                for i in 0..=x {
                    let register = Register::from_repr(i).unwrap();
                    self.registers
                        .set(register, self.memory.get(self.index + i as u16));
                }
            }
            Instruction::Or(register, register1) => todo!(),
            Instruction::Rand(register, _) => todo!(),
            Instruction::Assign(register, register1) => todo!(),
            Instruction::SetDelay(register) => self.delay_timer.set(self.registers.get(register)),
            Instruction::SetSound(register) => self.sound_timer.set(self.registers.get(register)),
            Instruction::ShiftLeft(register, register1) => todo!(),
            Instruction::ShiftRight(register, register1) => todo!(),
            Instruction::SkipIfKey(register) => todo!(),
            Instruction::SkipIfNotKey(register) => todo!(),
            Instruction::StoreMemory(x) => {
                let x = x as u8;
                for i in 0..=x {
                    let register = Register::from_repr(i).unwrap();
                    self.memory
                        .set(self.index + i as u16, self.registers.get(register));
                }
            }
            Instruction::Subtract(register, register1) => todo!(),
            Instruction::Xor(register, register1) => todo!(),
            Instruction::Call(_) => todo!(),
        }
        self.pc + 2
    }

    pub fn tick(&mut self) {
        let instruction = self.fetch();
        let instruction = Instruction::decode(instruction);
        self.pc = self.execute(instruction);
    }

    /// The caller should tick the timers at a 60hz frequency.
    pub fn tick_timers(&mut self) {
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

    fn set_keypad(&mut self, keypad: Keypad) {
        self.cpu.set_keypad(keypad);
    }

    fn tick(&mut self) -> Option<EmulatorState> {
        self.cpu.tick();
        if self.frame_counter.tick() {
            self.cpu.tick_timers();
            return Some(EmulatorState {
                beep: self.cpu.is_beep(),
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

type Keypad = [bool; 16];

struct GuiControls {
    keypad: Keypad,
}

struct DebuggerApp {
    state_rx: Receiver<EmulatorState>,
    ui_tx: SyncSender<Keypad>,
    screen: Screen,
    display_texture: egui::TextureHandle,
    _stream: OutputStream,
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
        let (state_tx, state_rx) = std::sync::mpsc::sync_channel(1);
        let (ui_tx, ui_rx) = std::sync::mpsc::sync_channel(1);
        let ctx = cc.egui_ctx.clone();
        let mut emulator = Emulator::new(rom);
        thread::spawn(move || {
            loop {
                let start = Instant::now();
                let ips = 700;
                let instruction_time = Duration::from_secs_f64(1.0 / ips as f64);
                if let Ok(keypad) = ui_rx.try_recv() {
                    emulator.set_keypad(keypad);
                }
                if let Some(state) = emulator.tick() {
                    if state_tx.send(state).is_err() {
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
            state_rx,
            ui_tx,
            screen: Screen::default(),
            display_texture,
            _stream: stream,
            sink,
        }
    }

    fn check_for_updates(&mut self) {
        if let Ok(state) = self.state_rx.try_recv() {
            self.screen = state.screen;
            if state.beep {
                self.sink.play();
            } else {
                self.sink.pause();
            }
        }
    }

    /// Check for and map CHIP-8 key presses
    fn check_keyboard(&self, ctx: &egui::Context) -> [bool; 16] {
        let mut keypad = [false; 16];
        ctx.input(|i| {
            // Map 1, 2, 3, 4 -> 1, 2, 3, C
            keypad[0x1] = i.key_down(Key::Num1);
            keypad[0x2] = i.key_down(Key::Num2);
            keypad[0x3] = i.key_down(Key::Num3);
            keypad[0xC] = i.key_down(Key::Num4);
            // Map Q, W, E, R -> 4, 5, 6, D
            keypad[0x4] = i.key_down(Key::Q);
            keypad[0x5] = i.key_down(Key::W);
            keypad[0x6] = i.key_down(Key::E);
            keypad[0xD] = i.key_down(Key::R);
            // Map A, S, D, F -> 7, 8, 9, E
            keypad[0x7] = i.key_down(Key::A);
            keypad[0x8] = i.key_down(Key::S);
            keypad[0x9] = i.key_down(Key::D);
            keypad[0xE] = i.key_down(Key::F);
            // Map Z, X, C, V -> A, 0, B, F
            keypad[0xA] = i.key_down(Key::Z);
            keypad[0x0] = i.key_down(Key::X);
            keypad[0xB] = i.key_down(Key::C);
            keypad[0xF] = i.key_down(Key::V);
        });
        keypad
    }

    fn render_content(&mut self, ui: &mut egui::Ui) {
        self.check_for_updates();
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
