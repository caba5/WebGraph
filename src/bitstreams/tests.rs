use std::{fs::{self, File}, rc::Rc};

use super::{BinaryWriterBuilder, BinaryReader, tables::{GAMMAS, ZETAS_3}};

fn write_unary(writer: &mut BinaryWriterBuilder, x: u64) -> u64 {
    if x < writer.free as u64 {
        return writer.write_in_current(1, x + 1);
    }

    let shift = writer.free;
    let x = x - shift as u64;

    writer.written_bits += shift;
    writer.write(writer.current);
    writer.free = 8;
    writer.current = 0;
    
    let mut i = x >> 3;

    writer.written_bits += x as usize & 0x7FFFFFF8;
    
    while i != 0 {
        writer.write(0);
        i -= 1;
    }

    writer.write_in_current(1, (x & 7) + 1);

    x + shift as u64 + 1
}

fn write_gamma(writer: &mut BinaryWriterBuilder, x: u64) -> u64 {
    // if x < MAX_PRECOMPUTED TODO

    let x = x + 1; // Code [0, +inf - 1]
    let msb = (u64::BITS - 1 - x.leading_zeros()) as u64;

    write_unary(writer, msb) + writer.push_bits(x, msb)
}

fn write_delta(writer: &mut BinaryWriterBuilder, x: u64) -> u64 {
    // if x < MAX_PRECOMPUTED TODO

    let x =  x + 1; // Code [0, +inf - 1]
    let msb = (u64::BITS - 1 - x.leading_zeros()) as u64;
    write_gamma(writer, msb) + writer.push_bits(x, msb)
}

fn write_zeta(writer: &mut BinaryWriterBuilder, x: u64, zk: u64) -> u64 {
    let x = x + 1;
    let msb = (u64::BITS - 1 - x.leading_zeros()) as u64;
    let h = msb / zk;
    let unary = write_unary(writer, h);
    let left = 1 << (h * zk);
    unary + 
        if x - left < left 
            {writer.push_bits(x - left, h * zk + zk - 1)}
        else 
            {writer.push_bits(x, h * zk + zk)}
}

fn read_unary(reader: &mut BinaryReader) -> u64 {
    if reader.fill < 16 {
        reader.refill();
    }

    let mut x = u32::leading_zeros((reader.current as u32) << (32 - reader.fill));
    if x < reader.fill as u32{
        reader.read_bits += x as usize + 1;
        reader.fill -= x as usize + 1;
        return x as u64;
    }

    x = reader.fill as u32;
    let mut read = reader.read();

    if read.is_ok() {
        reader.current = read.unwrap();
        while reader.current == 0 && read.is_ok() {
            x += 8;
            read = reader.read();
            if let Ok(r) = read {
                reader.current = r;
            }
        }
    }

    reader.fill = (63 - u64::leading_zeros(reader.current)) as usize;
    x += 7 - reader.fill as u32;
    reader.read_bits += x as usize + 1;
    x as u64
}

fn read_gamma(reader: &mut BinaryReader, use_table: bool) -> u64 {
    if use_table && (reader.fill >= 16 || reader.refill() >= 16) {
        let precomp = GAMMAS[reader.current as usize >> (reader.fill - 16) & 0xFFFF];

        if precomp.1 != 0 {
            reader.read_bits += precomp.1 as usize;
            reader.fill -= precomp.1 as usize;

            return precomp.0 as u64;
        }
    }
    
    let msb = read_unary(reader);
    ((1 << msb) | reader.read_int(msb)) - 1
}

fn read_delta(reader: &mut BinaryReader) -> u64 {
    let msb = read_gamma(reader, false);
    ((1 << msb) | reader.read_int(msb)) - 1
}

fn read_zeta(reader: &mut BinaryReader, zk: u64) -> u64 {
    if zk == 3 && (reader.fill >= 16 || reader.refill() >= 16) {
        let precomp = ZETAS_3[reader.current as usize >> (reader.fill - 16) & 0xFFFF];

        if precomp.1 != 0 {
            reader.read_bits += precomp.1 as usize;
            reader.fill -= precomp.1 as usize;

            return precomp.0 as u64;
        }
    }

    let unary = read_unary(reader);
    let left = 1 << (unary * zk);
    let m = reader.read_int(unary * zk + zk - 1);
    if m < left {m + left - 1} else {(m << 1) + reader.read_from_current(1) - 1}
}

fn test_correctness_write_and_read_to_file(code: &str) {
    let mut writer_builder = BinaryWriterBuilder::new();

    for x in 0..100000 {
        let _ = match code {
            "UNARY" => write_unary(&mut writer_builder, x),
            "GAMMA" => write_gamma(&mut writer_builder, x),
            "DELTA" => write_delta(&mut writer_builder, x),
            "ZETA" => write_zeta(&mut writer_builder, x, 3),
            _ => unreachable!()
        };
    }

    let written: Rc<[u8]> = writer_builder.build().os.into();

    fs::write(code, written).unwrap();
    
    let read_file = fs::read(code).unwrap();

    let mut binary_reader = BinaryReader::new(read_file.into());

    for x in 0..100000 {
        match code {
            "UNARY" => assert_eq!(read_unary(&mut binary_reader), x),
            "GAMMA" => assert_eq!(read_gamma(&mut binary_reader, false), x),
            "DELTA" => assert_eq!(read_delta(&mut binary_reader), x),
            "ZETA" => assert_eq!(read_zeta(&mut binary_reader, 3), x),
            _ => unreachable!()
        };
    }

    fs::remove_file(code).unwrap();
}

#[test]
fn test_unary() {
    test_correctness_write_and_read_to_file("UNARY");
}

#[test]
fn test_gamma() {
    test_correctness_write_and_read_to_file("GAMMA");
}

#[test]
fn test_delta() {
    test_correctness_write_and_read_to_file("DELTA");
}

#[test]
fn test_zeta() {
    test_correctness_write_and_read_to_file("ZETA");
}

#[test]
fn test_reposition() {
    let mut writer_builder = BinaryWriterBuilder::new();

    write_unary(&mut writer_builder, 10);
    write_unary(&mut writer_builder, 5);
    write_unary(&mut writer_builder, 5);

    let written = writer_builder.build().os.into();
    let mut binary_reader = BinaryReader::new(written);

    binary_reader.position(11);
    assert_eq!(read_unary(&mut binary_reader), 5);
    binary_reader.position(17);
    assert_eq!(read_unary(&mut binary_reader), 5);
    binary_reader.position(0);
    assert_eq!(read_unary(&mut binary_reader), 10);
}

#[test]
fn test_written_bits_number_correctness() {
    let mut writer_builder = BinaryWriterBuilder::new();

    write_unary(&mut writer_builder, 10);
    write_unary(&mut writer_builder, 5);
    write_unary(&mut writer_builder, 5);

    assert_eq!(writer_builder.written_bits, 23);
}

#[test]
fn test_reposition_over_64_bits() {
    let mut writer_builder = BinaryWriterBuilder::new();

    write_unary(&mut writer_builder, 32);
    write_unary(&mut writer_builder, 30);
    write_unary(&mut writer_builder, 31);
    write_unary(&mut writer_builder, 31);

    let written = writer_builder.build().os.into();
    let mut binary_reader = BinaryReader::new(written);

    binary_reader.position(33);
    assert_eq!(read_unary(&mut binary_reader), 30); 
    binary_reader.position(64);
    assert_eq!(read_unary(&mut binary_reader), 31);
    binary_reader.position(0);
    assert_eq!(read_unary(&mut binary_reader), 32);
    binary_reader.position(96);
    assert_eq!(read_unary(&mut binary_reader), 31);
}

#[test]
fn test_simple_integer_writing() {
    let mut writer_builder = BinaryWriterBuilder::new();

    writer_builder.push_bits(5, 3);
    writer_builder.push_bits(10, 4);
    writer_builder.push_bits(5, 3);

    let written = writer_builder.build().os.into();
    let mut binary_reader = BinaryReader::new(written);

    assert_eq!(binary_reader.read_int(3), 5);
    assert_eq!(binary_reader.read_int(4), 10);
    assert_eq!(binary_reader.read_int(3), 5);
}

#[test]
fn test_gamma_precomputed_table_correctness() {
    let mut writer_builder = BinaryWriterBuilder::new();

    write_gamma(&mut writer_builder, 1000);
    write_gamma(&mut writer_builder, 2000);
    write_gamma(&mut writer_builder, 0);
    write_gamma(&mut writer_builder, 30);
    write_gamma(&mut writer_builder, 256);
    write_gamma(&mut writer_builder, 999);
    write_gamma(&mut writer_builder, 40000);

    let written: Rc<[u8]> = writer_builder.build().os.into();
    let mut binary_reader_table = BinaryReader::new(written.clone());
    let mut binary_reader_normal = BinaryReader::new(written);

    assert_eq!(read_gamma(&mut binary_reader_table, true), 1000);
    assert_eq!(read_gamma(&mut binary_reader_normal, false), 1000);
    assert_eq!(read_gamma(&mut binary_reader_table, true), 2000);
    assert_eq!(read_gamma(&mut binary_reader_normal, false), 2000);
    assert_eq!(read_gamma(&mut binary_reader_table, true), 0);
    assert_eq!(read_gamma(&mut binary_reader_normal, false), 0);
    assert_eq!(read_gamma(&mut binary_reader_table, true), 30);
    assert_eq!(read_gamma(&mut binary_reader_normal, false), 30);
    assert_eq!(read_gamma(&mut binary_reader_table, true), 256);
    assert_eq!(read_gamma(&mut binary_reader_normal, false), 256);
    assert_eq!(read_gamma(&mut binary_reader_table, true), 999);
    assert_eq!(read_gamma(&mut binary_reader_normal, false), 999);
    assert_eq!(read_gamma(&mut binary_reader_table, true), 40000);
    assert_eq!(read_gamma(&mut binary_reader_normal, false), 40000);
}

#[test]
fn test_zeta_precomputed_table_correctness() {
    let mut writer_builder = BinaryWriterBuilder::new();

    write_zeta(&mut writer_builder, 1000, 3);
    write_zeta(&mut writer_builder, 2000, 3);
    write_zeta(&mut writer_builder, 0, 3);
    write_zeta(&mut writer_builder, 30, 3);
    write_zeta(&mut writer_builder, 256, 3);
    write_zeta(&mut writer_builder, 999, 3);
    write_zeta(&mut writer_builder, 40000, 3);

    let written = writer_builder.build().os.into();
    let mut binary_reader = BinaryReader::new(written);

    assert_eq!(read_zeta(&mut binary_reader, 3), 1000);
    assert_eq!(read_zeta(&mut binary_reader, 3), 2000);
    assert_eq!(read_zeta(&mut binary_reader, 3), 0);
    assert_eq!(read_zeta(&mut binary_reader, 3), 30);
    assert_eq!(read_zeta(&mut binary_reader, 3), 256);
    assert_eq!(read_zeta(&mut binary_reader, 3), 999);
    assert_eq!(read_zeta(&mut binary_reader, 3), 40000);
}

#[test]
fn test_read_lol() {
    let f = File::open("/home/caba/Desktop/zeta3.in.16").unwrap();
    let up_to = f.metadata().unwrap().len();

    let v = fs::read("/home/caba/Desktop/zeta3.in.16").unwrap();

    let mut final_v = Vec::new();

    // let mut reader = BinaryReader::new(v.into());

    let mut int: u32 = 0;

    eprint!("const ZETAS: &[(u16, u8)] = &[");
    for i in 0..up_to {
        int = (int << 8) | v[i as usize] as u32;
        if (i + 1) % 4 == 0 {
            eprint!("({}, {}), ", int & 0xFFFF, int >> 16);
            final_v.push(int);
            int = 0;
        }
    }
    eprintln!("];");

    assert!(final_v.len() == up_to as usize / 4);

    // println!("{:?}", final_v);

}