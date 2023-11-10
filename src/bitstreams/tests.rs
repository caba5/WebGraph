use std::{fs::{self, File}, rc::Rc};

use super::{BinaryWriter, BinaryReader, tables::{GAMMAS, ZETAS_3, DELTAS}};

fn write_unary(writer: &mut BinaryWriter, x: u64) -> u64 {
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

fn write_gamma(writer: &mut BinaryWriter, x: u64) -> u64 {
    // if x < MAX_PRECOMPUTED TODO

    let x = x + 1; // Code [0, +inf - 1]
    let msb = (u64::BITS - 1 - x.leading_zeros()) as u64;

    write_unary(writer, msb) + writer.push_bits(x, msb)
}

fn write_delta(writer: &mut BinaryWriter, x: u64) -> u64 {
    // if x < MAX_PRECOMPUTED TODO

    let x =  x + 1; // Code [0, +inf - 1]
    let msb = (u64::BITS - 1 - x.leading_zeros()) as u64;
    write_gamma(writer, msb) + writer.push_bits(x, msb)
}

fn write_zeta(writer: &mut BinaryWriter, x: u64, zk: u64) -> u64 {
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

fn read_delta(reader: &mut BinaryReader, use_table: bool) -> u64 {
    if use_table && (reader.fill >= 16 || reader.refill() >= 16) {
        let precomp = DELTAS[reader.current as usize >> (reader.fill - 16) & 0xFFFF];

        if precomp.1 != 0 {
            reader.read_bits += precomp.1 as usize;
            reader.fill -= precomp.1 as usize;

            return precomp.0 as u64;
        }
    }

    let msb = read_gamma(reader, false);
    ((1 << msb) | reader.read_int(msb)) - 1
}

fn read_zeta(reader: &mut BinaryReader, zk: u64, use_table: bool) -> u64 {
    if use_table && zk == 3 && (reader.fill >= 16 || reader.refill() >= 16) {
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
    let mut writer_builder = BinaryWriter::new();

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
            "DELTA" => assert_eq!(read_delta(&mut binary_reader, false), x),
            "ZETA" => assert_eq!(read_zeta(&mut binary_reader, 3, false), x),
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
    let mut writer_builder = BinaryWriter::new();

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
    let mut writer_builder = BinaryWriter::new();

    write_unary(&mut writer_builder, 10);
    write_unary(&mut writer_builder, 5);
    write_unary(&mut writer_builder, 5);

    assert_eq!(writer_builder.written_bits, 23);
}

#[test]
fn test_reposition_over_64_bits() {
    let mut writer_builder = BinaryWriter::new();

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
    let mut writer_builder = BinaryWriter::new();

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
    let mut writer_builder = BinaryWriter::new();

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
fn test_delta_precomputed_table_correctness() {
    let mut writer_builder = BinaryWriter::new();

    write_delta(&mut writer_builder, 1000);
    write_delta(&mut writer_builder, 2000);
    write_delta(&mut writer_builder, 0);
    write_delta(&mut writer_builder, 30);
    write_delta(&mut writer_builder, 256);
    write_delta(&mut writer_builder, 999);
    write_delta(&mut writer_builder, 40000);

    let written: Rc<[u8]> = writer_builder.build().os.into();
    let mut binary_reader_table = BinaryReader::new(written.clone());
    let mut binary_reader_normal = BinaryReader::new(written);

    assert_eq!(read_delta(&mut binary_reader_table, true), 1000);
    assert_eq!(read_delta(&mut binary_reader_normal, false), 1000);
    assert_eq!(read_delta(&mut binary_reader_table, true), 2000);
    assert_eq!(read_delta(&mut binary_reader_normal, false), 2000);
    assert_eq!(read_delta(&mut binary_reader_table, true), 0);
    assert_eq!(read_delta(&mut binary_reader_normal, false), 0);
    assert_eq!(read_delta(&mut binary_reader_table, true), 30);
    assert_eq!(read_delta(&mut binary_reader_normal, false), 30);
    assert_eq!(read_delta(&mut binary_reader_table, true), 256);
    assert_eq!(read_delta(&mut binary_reader_normal, false), 256);
    assert_eq!(read_delta(&mut binary_reader_table, true), 999);
    assert_eq!(read_delta(&mut binary_reader_normal, false), 999);
    assert_eq!(read_delta(&mut binary_reader_table, true), 40000);
    assert_eq!(read_delta(&mut binary_reader_normal, false), 40000);
}

#[test]
fn test_zeta_precomputed_table_correctness() {
    let mut writer_builder = BinaryWriter::new();

    write_zeta(&mut writer_builder, 1000, 3);
    write_zeta(&mut writer_builder, 2000, 3);
    write_zeta(&mut writer_builder, 0, 3);
    write_zeta(&mut writer_builder, 30, 3);
    write_zeta(&mut writer_builder, 256, 3);
    write_zeta(&mut writer_builder, 999, 3);
    write_zeta(&mut writer_builder, 40000, 3);

    let written: Rc<[u8]> = writer_builder.build().os.into();
    let mut binary_reader_table = BinaryReader::new(written.clone());
    let mut binary_reader_normal = BinaryReader::new(written);

    assert_eq!(read_zeta(&mut binary_reader_table, 3, true), 1000);
    assert_eq!(read_zeta(&mut binary_reader_normal, 3, false), 1000);
    assert_eq!(read_zeta(&mut binary_reader_table, 3, true), 2000);
    assert_eq!(read_zeta(&mut binary_reader_normal, 3, false), 2000);
    assert_eq!(read_zeta(&mut binary_reader_table, 3, true), 0);
    assert_eq!(read_zeta(&mut binary_reader_normal, 3, false), 0);
    assert_eq!(read_zeta(&mut binary_reader_table, 3, true), 30);
    assert_eq!(read_zeta(&mut binary_reader_normal, 3, false), 30);
    assert_eq!(read_zeta(&mut binary_reader_table, 3, true), 256);
    assert_eq!(read_zeta(&mut binary_reader_normal, 3, false), 256);
    assert_eq!(read_zeta(&mut binary_reader_table, 3, true), 999);
    assert_eq!(read_zeta(&mut binary_reader_normal, 3, false), 999);
    assert_eq!(read_zeta(&mut binary_reader_table, 3, true), 40000);
    assert_eq!(read_zeta(&mut binary_reader_normal, 3, false), 40000);
}