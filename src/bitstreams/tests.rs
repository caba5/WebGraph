use std::fs;

use super::BinaryWriterBuilder;

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

#[test]
fn test_correctness_write_and_read_to_file() {
    let mut write_builder = BinaryWriterBuilder::new();
    let mut written_bits = 0;

    for x in 0..1000000 {
        written_bits += write_gamma(&mut write_builder, x);
    }

    fs::write("temp", write_builder.build().os).unwrap();
    
}