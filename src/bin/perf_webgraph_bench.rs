use std::{time::Instant, fs::File, io::BufReader, hint::black_box};

use clap::Parser;

use rand::Rng;
use webgraph::traits::{SequentialGraph, RandomAccessGraph};
use webgraph_rust::{EncodingType, utils::encodings::{UniversalCode, GammaCode, UnaryCode, ZetaCode}, webgraph::bvgraph::{BVGraph, BVGraphBuilder}, ImmutableGraph, properties::Properties};


#[derive(Parser, Debug)]
struct WGArgs {
    /// Source basename
    source_name: String,
}

const N_QUERIES: usize = 1000000;
const N_RUNS: usize = 3;

fn gen_queries(n_queries: usize, range_size: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    (0..n_queries)
        .map(|_| rng.gen_range(0..range_size))
        .collect()
}

fn create_graph<
    InBlockCoding: UniversalCode,
    InBlockCountCoding: UniversalCode,
    InOudegreeCoding: UniversalCode,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InResidualCoding: UniversalCode
>(props: &Properties, in_name: &str) -> BVGraph<
    InBlockCoding,
    InBlockCountCoding,
    InOudegreeCoding,
    InOffsetCoding,
    InReferenceCoding,
    InResidualCoding,
    GammaCode,
    GammaCode,
    GammaCode,
    GammaCode,
    UnaryCode,
    ZetaCode
> {
    BVGraphBuilder::<
        InBlockCoding,
        InBlockCountCoding,
        InOudegreeCoding,
        InOffsetCoding,
        InReferenceCoding,
        InResidualCoding,
        GammaCode,
        GammaCode,
        GammaCode,
        GammaCode,
        UnaryCode,
        ZetaCode,
    >::new()
        .set_min_interval_len(props.min_interval_len)
        .set_max_ref_count(props.max_ref_count)
        .set_window_size(props.window_size)
        .set_zeta(props.zeta_k)
        .set_num_nodes(props.nodes)
        .set_num_edges(props.arcs)
        .load_graph(in_name)
        .load_offsets(in_name)
        .load_outdegrees()
        .build()
}

fn main() {
    let args = WGArgs::parse();
    
    let properties_file = File::open(format!("{}.properties", args.source_name));
    let properties_file = properties_file.unwrap_or_else(|_| panic!("Could not find {}.properties", args.source_name));
    let p = java_properties::read(BufReader::new(properties_file)).unwrap_or_else(|_| panic!("Failed parsing the properties file"));

    let props = Properties::from(p);

    assert!(props.nodes as u64 <= u64::MAX, "This version of WebGraph cannot handle graphs with {} (>=2^63) nodes", props.nodes);
    
    let mut my_graph = match (props.block_coding, props.block_count_coding, props.outdegree_coding, props.offset_coding, props.reference_coding, props.residual_coding) {
        (EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::GAMMA, EncodingType::UNARY, EncodingType::ZETA) => // Default case 
            create_graph::<GammaCode, GammaCode, GammaCode, GammaCode, UnaryCode, ZetaCode>(&props, &args.source_name),
        _ => panic!("Unexpected encoding types", )
    };

    let vigna_graph = webgraph::graph::bvgraph::load(args.source_name).unwrap();

    assert_eq!(my_graph.num_nodes(), vigna_graph.num_nodes());

    let queries = gen_queries(N_QUERIES, my_graph.num_nodes());

    // let total = Instant::now();
    // for _ in 0..N_RUNS {
    //     for &query in queries.iter() {
    //         let _ = black_box(my_graph.decode_list(query));
    //     }
    // }
    // let avg_query = (total.elapsed().as_nanos() as f64) / (N_QUERIES * N_RUNS) as f64;
    // println!("time per query for my_graph: {}ns", avg_query);

    let total = Instant::now();
    for _ in 0..N_RUNS {
        for &query in queries.iter() {
            let _ = black_box(vigna_graph.successors(query));
            // let _: Vec<usize> = s.collect();
        }
    }
    let avg_query = (total.elapsed().as_nanos() as f64) / (N_QUERIES * N_RUNS) as f64;
    println!("time per query for vigna_graph: {}ns", avg_query);

    let total = Instant::now();
    for _ in 0..N_RUNS {
        for &query in queries.iter() {
            let _ = black_box(my_graph.decode_list(query));
        }
    }
    let avg_query = (total.elapsed().as_nanos() as f64) / (N_QUERIES * N_RUNS) as f64;
    println!("time per query for my_graph: {}ns", avg_query);
}