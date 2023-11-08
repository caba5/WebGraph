use std::{time::Instant, fs::{File, self}, io::BufReader, cell::RefCell, rc::Rc};

use clap::Parser;
use rand::{Rng, seq::SliceRandom, thread_rng};
use webgraph_rust::{properties::Properties, webgraph::{bvgraph_huffman_in::BVGraphBuilder, bvgraph_huffman_out::{INTERVALS_LEN_IDX_BEGIN, INTERVALS_LEN_IDX_LEN}}, utils::encodings::{GammaCode, UnaryCode, ZetaCode, Huff}, EncodingType, ImmutableGraph, bitstreams::BinaryReader, huffman_zuckerli::huffman_decoder::HuffmanDecoder};

#[derive(Parser, Debug)]
#[command(about = "Generate a graph having the blocks, the intervals and the residuals Huffman-encoded")]
struct Args {
    /// The basename of the huffman-compressed graph file
    source_name: String,
    /// The destination basename of the graph file
    dest_name: String,
    /// Performance test
    #[arg(short, long = "perf", default_value_t = false)]
    perf_test: bool,
}

fn main() {
    let args = Args::parse();

    let properties_file = File::open(format!("{}.properties", args.source_name));
    let properties_file = properties_file.unwrap_or_else(|_| panic!("Could not find {}.properties", args.source_name));
    let p = java_properties::read(BufReader::new(properties_file)).unwrap_or_else(|_| panic!("Failed parsing the properties file"));
    let props = Properties::from(p);
    
    match (props.block_coding, props.block_count_coding, props.outdegree_coding, props.offset_coding, props.reference_coding, props.interval_coding, props.residual_coding) {
        (EncodingType::HUFFMAN, EncodingType::GAMMA, EncodingType::HUFFMAN, EncodingType::GAMMA, EncodingType::UNARY, EncodingType::HUFFMAN, EncodingType::HUFFMAN) => {},
        _ => panic!("The graph has to be Huffman-encoded (that is, its blocks, intervals, and residuals must be encoded by Huffman)")
    };

    let mut bvgraph = BVGraphBuilder::<
        Huff, GammaCode, Huff, GammaCode, UnaryCode, Huff, Huff,
        // Default encoding
        GammaCode, GammaCode, GammaCode, GammaCode, UnaryCode, GammaCode, ZetaCode
    >::new()
        .set_min_interval_len(props.min_interval_len)
        .set_max_ref_count(props.max_ref_count)
        .set_window_size(props.window_size)
        .set_zeta(props.zeta_k)
        .set_huff_outdegrees_parameters(props.huff_outdegrees_bits)
        .set_huff_blocks_parameters(props.huff_blocks_bits)
        .set_huff_residuals_parameters(props.huff_residuals_bits)
        .set_huff_intervals_left_parameters(props.huff_intervals_left_bits)
        .set_huff_intervals_len_parameters(props.huff_intervals_len_bits)
        .set_num_nodes(props.nodes)
        .set_num_edges(props.arcs)
        .load_graph(&args.source_name)
        .load_offsets(&args.source_name)
        .load_outdegrees()
        .build();

    let ibs = Rc::new(RefCell::new(BinaryReader::new(bvgraph.graph_memory.clone())));

    let mut huff_decoder = HuffmanDecoder::new();
    huff_decoder.decode_headers(&mut ibs.borrow_mut(), INTERVALS_LEN_IDX_BEGIN + INTERVALS_LEN_IDX_LEN);

    let n = bvgraph.num_nodes();

    for node in 0..n {
        bvgraph.decode_list(node, ibs.clone(), None, &mut [], &mut huff_decoder, true);
    }

    if args.perf_test {
        let mut shuffled_seq: Vec<usize> = (0..n).collect();
        shuffled_seq.shuffle(&mut thread_rng());

        for node in shuffled_seq.into_iter() {
            bvgraph.decode_list(node, ibs.clone(), None, &mut [], &mut huff_decoder, true);
        }
    } else {
        for node in 0..n {
            bvgraph.decode_list(node, ibs.clone(), None, &mut [], &mut huff_decoder, true);
        }
    }

    let mut out_stats = String::new();
        
    out_stats.push_str("################### Random-access Huffman decompression stats ###################\n");
    out_stats.push_str(&format!("time outdegrees {} ns\n", bvgraph.decompression_stats.borrow_mut().outdegree_time.total_time));
    out_stats.push_str(&format!("time blocks {} ns\n", bvgraph.decompression_stats.borrow_mut().block_time.total_time));
    out_stats.push_str(&format!("time block count {} ns\n", bvgraph.decompression_stats.borrow_mut().block_count_time.total_time));
    out_stats.push_str(&format!("time references {} ns\n", bvgraph.decompression_stats.borrow_mut().reference_time.total_time));
    out_stats.push_str(&format!("time interval count {} ns\n", bvgraph.decompression_stats.borrow_mut().interval_count_time.total_time));
    out_stats.push_str(&format!("time intervals {} ns\n", bvgraph.decompression_stats.borrow_mut().interval_time.total_time));
    out_stats.push_str(&format!("time residuals {} ns\n", bvgraph.decompression_stats.borrow_mut().residual_time.total_time));

    fs::write(format!("{}.stats", args.dest_name), out_stats).unwrap();
}