#![feature(iter_array_chunks)]
use eframe::egui::{self, Color32, Frame, Key, RichText, ScrollArea, SidePanel, TextStyle, Vec2};
use rand::prelude::*;
use rodio::{OutputStream, Sink, Source};
use spin_sleep::sleep;
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicU64};
use std::sync::mpsc::{Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};
use strum::IntoEnumIterator;

#[derive(Default, Debug, Clone)]
struct CPU {
    pc: u16,
    index: u16,
    stack: Vec<u16>,
    delay_timer: Timer,
    sound_timer: Timer,
    registers: Registers,
    memory: Memory,
    screen: Screen,
    keypad: Keypad,
}

impl CPU {
    fn new(rom: Vec<u8>, keypad: Keypad) -> Self {
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
            keypad,
            ..Default::default()
        }
    }

    fn is_beep(&self) -> bool {
        self.sound_timer.get() > 0
    }

    fn fetch(&self) -> u16 {
        let pc = self.pc as usize;
        (self.memory.0[pc] as u16) << 8 | self.memory.0[pc + 1] as u16
    }

    fn display(&mut self, x: Register, y: Register, height: u8) {
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

    fn execute(&mut self, instruction: Instruction) -> u16 {
        use Instruction::*;
        match instruction {
            Add(vx, val) => {
                *self.registers.get_mut(vx) = self.registers.get(vx).wrapping_add(val);
            }
            AddIndex(vx) => self.index = self.index.wrapping_add(self.registers.get(vx) as u16),
            AddReg(vx, vy) => {
                let vy = self.registers.get(vy);
                let vx = self.registers.get_mut(vx);
                let (val, overflow) = vx.overflowing_add(vy);
                *vx = val;
                self.registers.set(Register::VF, overflow as u8);
            }
            And(vx, vy) => *self.registers.get_mut(vx) &= self.registers.get(vy),
            Assign(vx, vy) => self.registers.set(vx, self.registers.get(vy)),
            BinaryDecimalConversion(vx) => {
                let val = self.registers.get(vx);
                let hundreds = val / 100;
                let tens = (val / 10) % 10;
                let ones = val % 10;
                self.memory.set(self.index, hundreds);
                self.memory.set(self.index + 1, tens);
                self.memory.set(self.index + 2, ones);
            }
            Call(_addr) => (),
            CallSubroutine(addr) => {
                self.stack.push(self.pc + 2);
                return addr;
            }
            CondSkip(cond) => {
                let cond = match cond {
                    Cond::Eq(vx, nn) => self.registers.get(vx) == nn,
                    Cond::Neq(vx, nn) => self.registers.get(vx) != nn,
                    Cond::EqReg(vx, vy) => self.registers.get(vx) == self.registers.get(vy),
                    Cond::NeqReg(vx, vy) => self.registers.get(vx) != self.registers.get(vy),
                };
                if cond {
                    return self.pc + 4;
                }
            }
            Display(x, y, height) => self.display(x, y, height),
            DisplayClear => self.screen = Screen::default(),
            FontCharacter(vx) => self.index = 0x50 + 5 * (self.registers.get(vx) & 0xF) as u16,
            GetDelay(vx) => self.registers.set(vx, self.delay_timer.0),
            GetKey(vx) => {
                let keys = self.keypad.get_state();
                for key in 0..16 {
                    if keys & (0b1 << key) > 0 {
                        self.registers.set(vx, key);
                        return self.pc + 2;
                    }
                }
                return self.pc;
            }
            Jump(addr) => return addr,
            JumpOffset(addr) => return addr.wrapping_add(self.registers.get(Register::V0) as u16),
            LoadMemory(x) => {
                let x = x as u8;
                for i in 0..=x {
                    let register = Register::from_repr(i).unwrap();
                    self.registers
                        .set(register, self.memory.get(self.index + i as u16));
                }
            }
            Or(vx, vy) => *self.registers.get_mut(vx) |= self.registers.get(vy),
            Rand(vx, nn) => self.registers.set(vx, rand::rng().random::<u8>() & nn),
            Return => return self.stack.pop().unwrap(),
            SetDelay(vx) => self.delay_timer.set(self.registers.get(vx)),
            SetIndex(val) => self.index = val,
            SetRegister(vx, val) => self.registers.set(vx, val),
            SetSound(vx) => self.sound_timer.set(self.registers.get(vx)),
            ShiftLeft(vx, _vy) => {
                let (val, overflow) = self.registers.get(vx).overflowing_mul(2);
                self.registers.set(vx, val);
                self.registers.set(Register::VF, overflow as u8);
            }
            ShiftRight(vx, _vy) => {
                let val = self.registers.get(vx);
                self.registers.set(vx, val >> 1);
                self.registers.set(Register::VF, val & 0b1);
            }
            SkipIfKey(vx) => {
                let key_index = self.registers.get(vx) & 0xF;
                if self.keypad.is_pressed(key_index) {
                    return self.pc + 4;
                }
            }
            SkipIfNotKey(vx) => {
                let key_index = self.registers.get(vx) & 0xF;
                if !self.keypad.is_pressed(key_index) {
                    return self.pc + 4;
                }
            }
            StoreMemory(x) => {
                let x = x as u8;
                for i in 0..=x {
                    let register = Register::from_repr(i).unwrap();
                    self.memory
                        .set(self.index + i as u16, self.registers.get(register));
                }
            }
            Subtract(vx, vy) => {
                let vx_val = self.registers.get(vx);
                let vy_val = self.registers.get(vy);
                let (val, underflow) = vx_val.overflowing_sub(vy_val);
                self.registers.set(vx, val);
                self.registers.set(Register::VF, !underflow as u8);
            }
            SubtractOther(vx, vy) => {
                let vx_val = self.registers.get(vx);
                let vy_val = self.registers.get(vy);
                let (val, underflow) = vy_val.overflowing_sub(vx_val);
                self.registers.set(vx, val);
                self.registers.set(Register::VF, !underflow as u8);
            }
            Xor(vx, vy) => *self.registers.get_mut(vx) ^= self.registers.get(vy),
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
}

#[derive(Debug, Clone)]
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
}

#[derive(Debug, Clone)]
struct Screen([[bool; 64]; 32]);

impl Default for Screen {
    fn default() -> Self {
        Self([[false; 64]; 32])
    }
}

#[derive(Default, Debug, Clone)]
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

#[derive(Debug, strum::FromRepr, Copy, Clone, strum::EnumIter, strum::Display)]
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

#[derive(Default, Debug, Clone)]
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
    /// Shifts VX to the left by 1, then sets VF to 1 if the most significant bit
    /// of VX prior to that shift was set, or to 0 if it was unset.
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
    /// VY is subtracted from VX. VF is set to 0 when there's an underflow, and 1 when there is not. (i.e. VF set to 1 if VX >= VY and 0 if not).
    Subtract(Register, Register),
    /// Sets VX to VY minus VX. VF is set to 0 when there's an underflow, and 1 when there is not. (i.e. VF set to 1 if VY >= VX).
    SubtractOther(Register, Register),
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
            0x8 if n == 0x7 => SubtractOther(x, y),
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
            _ => panic!("Unknown opcode: 0x{:04x}", opcode),
        }
    }
}

#[derive(Debug)]
struct Emulator {
    cpu: CPU,
    running: Arc<AtomicBool>,
    target_fps: Arc<AtomicU32>,
    cycle_accumulator: u32,
}

impl Emulator {
    fn load_rom(&mut self, rom: Vec<u8>) {
        self.cpu = CPU::new(rom, self.cpu.keypad.clone())
    }

    fn tick(&mut self, target_ips: u32) -> Option<CPU> {
        self.cpu.tick();

        let target_fps = self.target_fps.load(Relaxed);
        let target_ips = target_ips;
        self.cycle_accumulator += target_fps;
        if self.cycle_accumulator > target_ips {
            self.cycle_accumulator -= target_ips;
            self.cpu.tick_timers();
            return Some(self.cpu.clone());
        }
        None
    }
}

#[derive(Default, Debug, Clone)]
struct Keypad(pub Arc<AtomicU16>);

impl Keypad {
    pub fn is_pressed(&self, key_index: u8) -> bool {
        if key_index > 15 {
            return false; // Or panic!("Key index out of bounds")
        }
        (self.get_state() >> key_index) & 1 == 1
    }

    pub fn get_state(&self) -> u16 {
        self.0.load(Relaxed)
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
    fn new(cc: &eframe::CreationContext, rom: Vec<u8>) -> Self {
        let (state_tx, state_rx) = std::sync::mpsc::sync_channel(1);
        let (rom_tx, rom_rx) = std::sync::mpsc::channel::<Vec<u8>>();
        let keypad = Keypad::default();
        let ctx = cc.egui_ctx.clone();
        ctx.set_visuals(egui::Visuals::dark());

        let running = Arc::new(AtomicBool::new(true));
        let target_ips = Arc::new(AtomicU32::new(700));
        let target_fps = Arc::new(AtomicU32::new(60));
        let instruction_counter: Arc<AtomicU64> = Arc::default();

        let cpu = CPU::new(rom, keypad.clone());
        let mut emulator = Emulator {
            cpu: cpu.clone(),
            running: running.clone(),
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
                }

                if let Ok(new_rom) = rom_rx.try_recv() {
                    emulator.load_rom(new_rom);
                    emulator.cycle_accumulator = 0;
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
        ui.heading("Register Viewer");
        egui::Grid::new("register_grid")
            .num_columns(4)
            .spacing([10.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                for row in Register::iter().array_chunks::<2>() {
                    for reg in row {
                        ui.label(RichText::new(format!("{reg}")).text_style(TextStyle::Monospace));
                        ui.label(
                            RichText::new(format!("0x{:02X}", cpu.registers.get(reg)))
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
                ui.label(RichText::new(format!("0x{:04X}", cpu.pc)));
                ui.label(RichText::new(format!("0x{:04X}", cpu.pc)));

                ui.end_row();

                ui.label(RichText::new("IR"));
                ui.label(RichText::new(format!("0x{:04X}", cpu.fetch())));
                ui.end_row();

                ui.label(RichText::new("I"));
                ui.label(RichText::new(format!("0x{:04X}", cpu.index)));
                ui.end_row();

                ui.label(RichText::new("Delay"));
                ui.label(RichText::new(format!("{}", cpu.delay_timer.get())));
                ui.end_row();

                ui.label(RichText::new("Sound"));
                ui.label(RichText::new(format!("{}", cpu.sound_timer.get())));
                ui.end_row();
            });
    }

    /// Renders the CPU stack
    fn render_stack_viewer(&self, ui: &mut egui::Ui, cpu: &CPU) {
        ui.heading("Stack Viewer");
        ScrollArea::vertical().max_height(150.0).show(ui, |ui| {
            egui::Grid::new("stack_grid")
                .num_columns(2)
                .striped(true)
                .show(ui, |ui| {
                    ui.label(RichText::new("Depth").strong());
                    ui.label(RichText::new("Contents").strong());
                    ui.end_row();

                    for (i, &addr) in cpu.stack.iter().enumerate().rev() {
                        ui.label(format!("{}", i));
                        ui.label(
                            RichText::new(format!("0x{:04X}", addr))
                                .text_style(TextStyle::Monospace),
                        );
                        ui.end_row();
                    }
                });
        });
    }

    fn render_memory_viewer(&self, ui: &mut egui::Ui, cpu: &CPU) {
        ui.heading("Memory Viewer");
        ScrollArea::vertical().show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.style_mut().override_text_style = Some(TextStyle::Monospace);
                ui.spacing_mut().item_spacing.x = 2.0;

                for (i, chunk) in cpu.memory.0.chunks(16).enumerate() {
                    let addr = i * 16;
                    let color = if (addr..addr + 16).contains(&(cpu.pc as usize)) {
                        Color32::YELLOW
                    } else {
                        Color32::LIGHT_GREEN
                    };

                    ui.label(RichText::new(format!("0x{:04X}", addr)).color(color));
                    ui.add(egui::Separator::default().vertical().shrink(10.0));
                    for byte in chunk {
                        ui.label(RichText::new(format!("{:02X}", byte)).color(color));
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
    }

    fn render_settings_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Settings");
        if ui.button("Load ROM...").clicked() {
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

        ui.separator();
        ui.label("Emulator");
        let mut ips = self.target_ips.load(Relaxed);
        ui.add(egui::Slider::new(&mut ips, 1..=100_000).text("Target IPS"));
        self.target_ips.store(ips, Relaxed);

        const FPS_OPTIONS: &[u32] = &[30, 60, 120, 144, 240];
        let mut fps = self.target_fps.load(Relaxed);
        ui.horizontal(|ui| {
            ui.label("Emulator Target FPS:");
            egui::ComboBox::new("fps_select", "")
                .selected_text(format!("{} FPS", fps))
                .show_ui(ui, |ui| {
                    // 3. Iterate over your options
                    for &fps_option in FPS_OPTIONS {
                        ui.selectable_value(&mut fps, fps_option, format!("{} FPS", fps_option));
                    }
                });
            self.target_fps.store(fps, Relaxed);
        });

        let running = self.running.load(Relaxed);
        if ui.button(if running { "Pause" } else { "Run" }).clicked() {
            self.running.store(!running, Relaxed);
        }

        ui.separator();
        ui.label("Display");
        ui.add(egui::Slider::new(&mut self.game_scale, 1.0..=32.0).text("Game Scale"));
        ui.horizontal(|ui| {
            ui.label(RichText::new("On Pixel: ").text_style(TextStyle::Monospace));
            ui.color_edit_button_srgba(&mut self.on_pixel_color);
        });
        ui.horizontal(|ui| {
            ui.label(RichText::new("Off Pixel:").text_style(TextStyle::Monospace));
            ui.color_edit_button_srgba(&mut self.off_pixel_color);
        });
    }

    fn render_info_panel(&self, ui: &mut egui::Ui) {
        ui.heading("Emulator Info");

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
            ui.label(
                RichText::new(if self.sink.is_paused() { "OK" } else { "BEEP" })
                    .color(Color32::LIGHT_GREEN),
            );
            ui.end_row();
        });
    }

    fn render_game_screen(&mut self, ui: &mut egui::Ui) {
        let screen = &self.last_state.screen;
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
    let rom = std::env::args().nth(1).map_or_else(Vec::new, |path| {
        std::fs::read(path).expect("Failed to read ROM from path")
    });
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
