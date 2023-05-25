use std::{env, process};

use webgraph_rust::webgraph::{EncodingType, BVGraph};

fn main() {  // TODO: CLAP
    if env::args().len() != 9 {
        println!("Wrong arguments. You need to specify:
            compression-type [Delta (1), Gamma (2), Golomb (3), Skewed Golomb (4), Unary (5), Zeta (6), Nibble (7)]
            window-size
            max-ref-count
            min-interval-length
            zetak
            store-outdeg: true (1) / false (0)
            source-name
            dest-name"
        );
        process::exit(0);
    }

    let mut arguments = env::args().into_iter();

    let comp = match arguments.next().unwrap().parse() {
        Ok(1) => EncodingType::DELTA,
        Ok(2) => EncodingType::GAMMA,
        Ok(3) => EncodingType::GOLOMB,
        Ok(4) => EncodingType::SKEWEDGOLOMB,
        Ok(5) => EncodingType::UNARY,
        Ok(6) => EncodingType::ZETA,
        Ok(7) => EncodingType::NIBBLE,
        _ => EncodingType::ZETA, // Default compression is zetak
    };

    arguments.next();

    let window_size = arguments.next().unwrap().parse().unwrap();
    let window_size = if window_size >= 0 {window_size} else {0}; // 0 = not used

    let max_ref_count = arguments.next().unwrap().parse().unwrap();
    let max_ref_count = if max_ref_count >= 0 {max_ref_count} else {0}; // 0 = not used

    let min_interval_length = arguments.next().unwrap().parse().unwrap();
    let min_interval_length = if min_interval_length >= 2 {min_interval_length} else {0}; // Interval len is either 2 either not used (0)

    let zetak = arguments.next().unwrap().parse().unwrap();
    let zetak = if comp == EncodingType::ZETA && zetak >= 0 {zetak} else if comp == EncodingType::ZETA {3} else {0}; // 3 is the default k value, 0 = not used

    // Probably can be removed
    let store_outdeg: i32 = arguments.next().unwrap().parse().unwrap();
    let store_outdeg = if store_outdeg == 0 {0} else {1}; // 0 as default
    
    let source_name = arguments.next().unwrap();

    let dest_name = arguments.next().unwrap();

    let graph = BVGraph::load(&source_name); // Loading the graph

    // ...ops...

    graph.store(&dest_name); // Storing the graph

}