import argparse
import sys
import onnx
import numpy as np
from onnx import helper, numpy_helper

def get_node_by_output(nodes, name):
    for node in nodes:
        if name in node.output:
            return node
    return None

def find_consumers(nodes, name):
    consumers = []
    for node in nodes:
        if name in node.input:
            consumers.append(node)
    return consumers

def get_initializer(graph, name):
    for init in graph.initializer:
        if init.name == name:
            return init
    return None

def convert_qdq_to_qop(model_path, output_path):
    print(f"Loading model from {model_path}...")
    model = onnx.load(model_path)
    graph = model.graph

    new_nodes = []
    outputs_to_remove = set()
    
    new_initializers = []
    
    # 0 for DequantizeLinear after ConvInteger/MatMulInteger
    zero_zp_init = numpy_helper.from_array(np.array(0, dtype=np.int32), "zero_zp_int32")
    new_initializers.append(zero_zp_init)

    # Find all Conv nodes
    for node in graph.node:
        if node.op_type == 'Conv':
            # Check if inputs are DequantizeLinear
            x_name = node.input[0]
            w_name = node.input[1]
            b_name = node.input[2] if len(node.input) > 2 else ""

            x_dq = get_node_by_output(graph.node, x_name)
            w_dq = get_node_by_output(graph.node, w_name)
            
            if not (x_dq and w_dq and x_dq.op_type == 'DequantizeLinear' and w_dq.op_type == 'DequantizeLinear'):
                print(f"Skipping {node.name}: inputs are not DequantizeLinear. x_dq: {x_dq.op_type if x_dq else 'None'}, w_dq: {w_dq.op_type if w_dq else 'None'}")
                continue
            
            # Check if output is consumed by QuantizeLinear
            y_name = node.output[0]
            consumers = find_consumers(graph.node, y_name)
            
            # Get the quantized inputs
            x_q_name = x_dq.input[0]
            x_scale_name = x_dq.input[1]
            x_zp = x_dq.input[2] if len(x_dq.input) > 2 else ""
            
            w_q_name = w_dq.input[0]
            w_scale_name = w_dq.input[1]
            w_zp = w_dq.input[2] if len(w_dq.input) > 2 else ""

            if len(consumers) == 1 and consumers[0].op_type == 'QuantizeLinear':
                # DequantizeLinear (x2) -> Conv -> QuantizeLinear  =>  QLinearConv
                y_q = consumers[0]
                y_scale = y_q.input[1]
                y_zp = y_q.input[2] if len(y_q.input) > 2 else ""
                q_conv_output = y_q.output[0]

                inputs = [
                    x_q_name, x_scale_name, x_zp,
                    w_q_name, w_scale_name, w_zp,
                    y_scale, y_zp
                ]
                if b_name:
                    # Quantize bias to int32
                    b_init = get_initializer(graph, b_name)
                    xs_init = get_initializer(graph, x_scale_name)
                    ws_init = get_initializer(graph, w_scale_name)
                    
                    if b_init and xs_init and ws_init:
                        b_val = numpy_helper.to_array(b_init)
                        xs_val = numpy_helper.to_array(xs_init)
                        ws_val = numpy_helper.to_array(ws_init)
                        
                        # QLinearConv expects bias to be 1D [C_out]
                        bq_val = np.round(b_val.flatten() / (xs_val * ws_val)).astype(np.int32)
                        bq_name = b_name + "_quantized"
                        bq_init = numpy_helper.from_array(bq_val, bq_name)
                        new_initializers.append(bq_init)
                        inputs.append(bq_name)
                    else:
                        print(f"Warning: Could not quantize bias for {node.name}, using original bias name. This may fail.")
                        inputs.append(b_name)

                new_node = helper.make_node(
                    'QLinearConv',
                    inputs=inputs,
                    outputs=[q_conv_output],
                    name=node.name + "_quant",
                    **{a.name: helper.get_attribute_value(a) for a in node.attribute}
                )
                new_nodes.append(new_node)
                outputs_to_remove.add(node.output[0])
                outputs_to_remove.add(x_dq.output[0])
                outputs_to_remove.add(w_dq.output[0])
                outputs_to_remove.add(y_q.output[0])
                print(f"Replaced {node.name} (Conv) with QLinearConv")

            else:
                # DequantizeLinear (x2) -> Conv  (no following QuantizeLinear)  =>  ConvInteger + Scaling
                inputs = [x_q_name, w_q_name]
                if x_zp:
                    inputs.append(x_zp)
                if w_zp:
                    if not x_zp:
                        inputs.append("")
                    inputs.append(w_zp)

                conv_int_out = node.name + "_int_output"
                new_node = helper.make_node(
                    'ConvInteger',
                    inputs=inputs,
                    outputs=[conv_int_out],
                    name=node.name + "_integer",
                    **{a.name: helper.get_attribute_value(a) for a in node.attribute}
                )
                new_nodes.append(new_node)
                
                # Add DequantizeLinear to restore scale
                xs_init = get_initializer(graph, x_scale_name)
                ws_init = get_initializer(graph, w_scale_name)
                
                if xs_init and ws_init:
                    xs_val = numpy_helper.to_array(xs_init)
                    ws_val = numpy_helper.to_array(ws_init)
                    comb_s_val = xs_val * ws_val
                    comb_s_name = node.name + "_combined_scale"
                    comb_s_init = numpy_helper.from_array(comb_s_val.astype(np.float32), comb_s_name)
                    new_initializers.append(comb_s_init)
                    
                    dq_out = node.name + "_dq_output"
                    # If there's a bias, we'll add it after DQ
                    # If no bias, DQ output is the final node output
                    final_node_out = node.output[0] if not b_name else dq_out
                    
                    dq_node = helper.make_node(
                        'DequantizeLinear',
                        inputs=[conv_int_out, comb_s_name, "zero_zp_int32"],
                        outputs=[final_node_out],
                        name=node.name + "_dq"
                    )
                    new_nodes.append(dq_node)
                    
                    if b_name:
                        # For Add to broadcast correctly [N, C, H, W] + [1, C, 1, 1]
                        # We need to reshape the bias if it's 1D
                        b_init = get_initializer(graph, b_name)
                        if b_init:
                            b_val = numpy_helper.to_array(b_init)
                            if b_val.ndim == 1:
                                # Get Conv attributes to determine rank
                                # Default is 2D conv (4D tensor)
                                # But we can check if it has 'strides' or other attrs
                                # For now assume 2D conv if not specified, or just use rank of conv output if we knew it
                                # Use the weight initializer shape to determine rank
                                # w_q_name was found earlier (w_dq.input[0])
                                w_init = get_initializer(graph, w_q_name)
                                if w_init:
                                    w_rank = len(w_init.dims)
                                    new_shape = [1] * w_rank
                                    new_shape[1] = b_val.shape[0]
                                    b_reshaped_val = b_val.reshape(new_shape)
                                    b_reshaped_name = b_name + "_reshaped"
                                    b_reshaped_init = numpy_helper.from_array(b_reshaped_val, b_reshaped_name)
                                    new_initializers.append(b_reshaped_init)
                                    bias_to_add = b_reshaped_name
                                else:
                                    bias_to_add = b_name
                            else:
                                bias_to_add = b_name
                        else:
                            bias_to_add = b_name

                        add_node = helper.make_node(
                            'Add',
                            inputs=[dq_out, bias_to_add],
                            outputs=[node.output[0]],
                            name=node.name + "_bias_add"
                        )
                        new_nodes.append(add_node)
                else:
                    print(f"Warning: Could not find scales for {node.name}, ConvInteger output will be unscaled!")
                    # Just rename output of ConvInteger to original output
                    new_node.output[0] = node.output[0]

                outputs_to_remove.add(node.output[0])
                outputs_to_remove.add(x_dq.output[0])
                outputs_to_remove.add(w_dq.output[0])
                print(f"Replaced {node.name} (Conv) with ConvInteger + scaling")

        elif node.op_type == 'MatMul':
            x_name = node.input[0]
            w_name = node.input[1]
            
            x_dq = get_node_by_output(graph.node, x_name)
            w_dq = get_node_by_output(graph.node, w_name)
            
            if not (x_dq and w_dq and x_dq.op_type == 'DequantizeLinear' and w_dq.op_type == 'DequantizeLinear'):
                continue

            y_name = node.output[0]
            consumers = find_consumers(graph.node, y_name)

            # Get the quantized inputs
            x_q_name = x_dq.input[0]
            x_scale_name = x_dq.input[1]
            x_zp = x_dq.input[2] if len(x_dq.input) > 2 else ""

            w_q_name = w_dq.input[0]
            w_scale_name = w_dq.input[1]
            w_zp = w_dq.input[2] if len(w_dq.input) > 2 else ""

            if len(consumers) == 1 and consumers[0].op_type == 'QuantizeLinear':
                # DequantizeLinear (x2) -> MatMul -> QuantizeLinear  =>  QLinearMatMul
                y_q = consumers[0]
                y_scale = y_q.input[1]
                y_zp = y_q.input[2] if len(y_q.input) > 2 else ""
                q_matmul_output = y_q.output[0]

                inputs = [
                    x_q_name, x_scale_name, x_zp,
                    w_q_name, w_scale_name, w_zp,
                    y_scale, y_zp
                ]

                new_node = helper.make_node(
                    'QLinearMatMul',
                    inputs=inputs,
                    outputs=[q_matmul_output],
                    name=node.name + "_quant",
                    **{a.name: helper.get_attribute_value(a) for a in node.attribute}
                )
                new_nodes.append(new_node)
                outputs_to_remove.add(node.output[0])
                outputs_to_remove.add(x_dq.output[0])
                outputs_to_remove.add(w_dq.output[0])
                outputs_to_remove.add(y_q.output[0])
                print(f"Replaced {node.name} (MatMul) with QLinearMatMul")

            else:
                # DequantizeLinear (x2) -> MatMul  (no following QuantizeLinear)  =>  MatMulInteger + Scaling
                inputs = [x_q_name, w_q_name]
                if x_zp:
                    inputs.append(x_zp)
                if w_zp:
                    if not x_zp:
                        inputs.append("")  # pad x_zp slot
                    inputs.append(w_zp)

                matmul_int_out = node.name + "_int_output"
                new_node = helper.make_node(
                    'MatMulInteger',
                    inputs=inputs,
                    outputs=[matmul_int_out],
                    name=node.name + "_integer",
                    **{a.name: helper.get_attribute_value(a) for a in node.attribute}
                )
                new_nodes.append(new_node)
                
                # Add DequantizeLinear to restore scale
                xs_init = get_initializer(graph, x_scale_name)
                ws_init = get_initializer(graph, w_scale_name)
                
                if xs_init and ws_init:
                    xs_val = numpy_helper.to_array(xs_init)
                    ws_val = numpy_helper.to_array(ws_init)
                    comb_s_val = xs_val * ws_val
                    comb_s_name = node.name + "_combined_scale"
                    comb_s_init = numpy_helper.from_array(comb_s_val.astype(np.float32), comb_s_name)
                    new_initializers.append(comb_s_init)
                    
                    dq_node = helper.make_node(
                        'DequantizeLinear',
                        inputs=[matmul_int_out, comb_s_name, "zero_zp_int32"],
                        outputs=[node.output[0]],
                        name=node.name + "_dq"
                    )
                    new_nodes.append(dq_node)
                else:
                    print(f"Warning: Could not find scales for {node.name}, MatMulInteger output will be unscaled!")
                    new_node.output[0] = node.output[0]

                outputs_to_remove.add(node.output[0])
                outputs_to_remove.add(x_dq.output[0])
                outputs_to_remove.add(w_dq.output[0])
                print(f"Replaced {node.name} (MatMul) with MatMulInteger + scaling")

    if not new_nodes:
        print("No QDQ patterns found to replace.")
        return
        
    final_nodes = []
    for node in graph.node:
        if node.output[0] not in outputs_to_remove:
            final_nodes.append(node)
            
    final_nodes.extend(new_nodes)
    
    # Topological sort
    ready_tensors = set([i.name for i in graph.input])
    ready_tensors.update([i.name for i in graph.initializer])
    
    unsorted_nodes = final_nodes.copy()
    sorted_nodes = []
    
    progress = True
    while unsorted_nodes and progress:
        progress = False
        remaining = []
        for node in unsorted_nodes:
            if all(inp in ready_tensors or inp == "" for inp in node.input):
                sorted_nodes.append(node)
                for out in node.output:
                    ready_tensors.add(out)
                progress = True
            else:
                remaining.append(node)
        unsorted_nodes = remaining
        
    if unsorted_nodes:
        print("Warning: Graph may not be fully connected or has cycles!")
        sorted_nodes.extend(unsorted_nodes)
    
    # Create new graph
    new_initializers_total = list(graph.initializer) + new_initializers
    
    new_graph = helper.make_graph(
        sorted_nodes,
        graph.name,
        graph.input,
        graph.output,
        new_initializers_total,
        None,
        graph.value_info
    )
    
    new_model = helper.make_model(new_graph, producer_name='qdq-to-qop-converter', opset_imports=model.opset_import)
    
    print(f"Saving QOp model to {output_path}...")
    onnx.save(new_model, output_path)
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description="Convert ONNX QDQ format to QOp format")
    parser.add_argument("input_onnx", help="Input ONNX model in QDQ format")
    parser.add_argument("output_onnx", help="Output ONNX model path in QOp format")
    args = parser.parse_args()
    
    convert_qdq_to_qop(args.input_onnx, args.output_onnx)


if __name__ == "__main__":
    main()
