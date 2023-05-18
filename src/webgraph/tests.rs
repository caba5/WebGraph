use super::*;

use serde::{Serialize, Deserialize};

#[test]
fn test_serialization() {
    let gr = BVGraph::new();
    gr.store("graph");
    let retrieved = BVGraph::load("graph");
    // println!("{:?}", gr);
    // println!("{:?}", retrieved);
    assert_eq!(retrieved, gr);
}