use onnxsim::{simplify_bytes, simplify_file, SimplifyOptions};

#[test]
fn test_init_env() {
    onnxsim::init_env();
}

#[test]
fn test_simplify_options() {
    let options = SimplifyOptions::new()
        .with_constant_folding(true)
        .with_shape_inference(true)
        .with_tensor_size_threshold(1024);

    assert!(options.constant_folding);
    assert!(options.shape_inference);
    assert_eq!(options.tensor_size_threshold, 1024);
}

#[test]
fn test_simplify_bytes_with_invalid_input() {
    // Test with invalid ONNX model bytes
    let invalid_model = b"not a valid onnx model";

    onnxsim::init_env();
    let result = simplify_bytes(invalid_model, SimplifyOptions::default());

    // Should fail with parse error
    assert!(result.is_err());
    match result {
        Err(onnxsim::OnnxSimError::ParseFailed(_)) => (),
        _ => panic!("Expected ParseFailed error"),
    }
}

#[test]
fn test_simplify_file_with_nonexistent_input() {
    let temp_dir = std::env::temp_dir();
    let input_path = temp_dir.join("nonexistent.onnx");
    let output_path = temp_dir.join("output.onnx");

    onnxsim::init_env();
    let result = simplify_file(&input_path, &output_path, SimplifyOptions::default());

    // Should fail with error
    assert!(result.is_err());
}
