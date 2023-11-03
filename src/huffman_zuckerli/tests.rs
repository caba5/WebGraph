use crate::bitstreams::{BinaryWriterBuilder, BinaryReader};

use super::{huffman_encoder::HuffmanEncoder, huffman_decoder::HuffmanDecoder};

#[test]
fn test_single_context_sequential() {
    let ints = vec![
        vec![100, 200, 3, 0, 10, 3, 2500000000]
    ];
    let num_contexts = ints.len();

    let mut writer = BinaryWriterBuilder::new();

    let mut huff_encoder = HuffmanEncoder::new();
    huff_encoder.init(&ints , &mut writer);

    for (ctx, int_arr) in ints.iter().enumerate() {
        for int in int_arr {
            huff_encoder.write_next(*int, &mut writer, ctx);
        }
    }

    let mut reader = BinaryReader::new(writer.build().os.into());

    let mut huff_decoder = HuffmanDecoder::new();

    huff_decoder.decode_headers(&mut reader, num_contexts);

    for (ctx, int) in ints.iter().enumerate() {
        for x in int {
            assert_eq!(*x, huff_decoder.read_next(&mut reader, ctx));
        }
    }
}

#[test]
fn test_multi_context_sequential() {
    let ints = vec![
        vec![100, 200, 3, 0, 10, 3, 2500000000],
        vec![1, 1, 1, 0, 10, 1, 0, 0, 1, 1],
        vec![0],
        vec![],
        vec![1111, 1111, 1111, 1111],
        vec![000, 1111, 000, 1111],
    ];
    let num_contexts = ints.len();

    let mut writer = BinaryWriterBuilder::new();

    let mut huff_encoder = HuffmanEncoder::new();
    huff_encoder.init(&ints , &mut writer);

    for (ctx, int_arr) in ints.iter().enumerate() {
        for int in int_arr {
            huff_encoder.write_next(*int, &mut writer, ctx);
        }
    }

    let mut reader = BinaryReader::new(writer.build().os.into());

    let mut huff_decoder = HuffmanDecoder::new();

    huff_decoder.decode_headers(&mut reader, num_contexts);

    for (ctx, int) in ints.iter().enumerate() {
        for x in int {
            assert_eq!(*x, huff_decoder.read_next(&mut reader, ctx));
        }
    }
}

#[test]
fn test_single_context_alternate() {
    let ints = vec![
        vec![100, 200, 3, 0, 10, 3, 2500000000]
    ];
    let num_contexts = ints.len();

    let mut writer = BinaryWriterBuilder::new();

    let mut huff_encoder = HuffmanEncoder::new();
    huff_encoder.init(&ints , &mut writer);

    huff_encoder.write_next(ints[0][2], &mut writer, 0);
    huff_encoder.write_next(ints[0][1], &mut writer, 0);
    huff_encoder.write_next(ints[0][3], &mut writer, 0);
    huff_encoder.write_next(ints[0][5], &mut writer, 0);
    huff_encoder.write_next(ints[0][5], &mut writer, 0);
    huff_encoder.write_next(ints[0][4], &mut writer, 0);
    huff_encoder.write_next(ints[0][2], &mut writer, 0);
    huff_encoder.write_next(ints[0][0], &mut writer, 0);

    let mut reader = BinaryReader::new(writer.build().os.into());

    let mut huff_decoder = HuffmanDecoder::new();

    huff_decoder.decode_headers(&mut reader, num_contexts);
    
    assert_eq!(ints[0][2], huff_decoder.read_next(&mut reader, 0));
    assert_eq!(ints[0][1], huff_decoder.read_next(&mut reader, 0));
    assert_eq!(ints[0][3], huff_decoder.read_next(&mut reader, 0));
    assert_eq!(ints[0][5], huff_decoder.read_next(&mut reader, 0));
    assert_eq!(ints[0][5], huff_decoder.read_next(&mut reader, 0));
    assert_eq!(ints[0][4], huff_decoder.read_next(&mut reader, 0));
    assert_eq!(ints[0][2], huff_decoder.read_next(&mut reader, 0));
    assert_eq!(ints[0][0], huff_decoder.read_next(&mut reader, 0));    
}

#[test]
fn test_multi_context_alternate() {
    let ints = vec![
        vec![100, 100, 3, 0],
        vec![1, 1, 1, 1],
        vec![0, 1, 100, 1],
        vec![1111, 1111, 1111, 1111],
        vec![000, 1111, 000, 1111],
    ];
    let num_contexts = ints.len();

    let mut writer = BinaryWriterBuilder::new();

    let mut huff_encoder = HuffmanEncoder::new();
    huff_encoder.init(&ints , &mut writer);

    for i in 0..ints[0].len() {
        for (ctx, nums) in ints.iter().enumerate() {
            huff_encoder.write_next(nums[i], &mut writer, ctx);
        }
    }

    let mut reader = BinaryReader::new(writer.build().os.into());

    let mut huff_decoder = HuffmanDecoder::new();

    huff_decoder.decode_headers(&mut reader, num_contexts);

    for i in 0..ints[0].len() {
        for (ctx, int) in ints.iter().enumerate() {
            assert_eq!(int[i], huff_decoder.read_next(&mut reader, ctx));
        }
    }
}

#[test]
#[should_panic]
fn test_non_existing_code() {
    let ints = vec![
        vec![100, 200, 3, 0, 10, 3, 2500000000]
    ];

    let mut writer = BinaryWriterBuilder::new();

    let mut huff_encoder = HuffmanEncoder::new();
    huff_encoder.init(&ints , &mut writer);

    huff_encoder.write_next(30, &mut writer, 0);   
}