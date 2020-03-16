from os import path
import onnx
from onnx import optimizer


def optimize_quantized_mobilenet(name_in, name_out):  #pylint: disable=too-many-locals
    """Optimize a quantized MobileNetV1 model for inference.

        - Remove the Pad node before the last AveragePool node.
        - Replace the Constant, Reshape and Gemm nodes after the last AveragePool node with
            an equivalent 1x1 Conv node.
    Args:
        - name_in (str): Path to the ONNX file containing the original MobileNetV1 graph.
        - name_out (str): Path to which the optimized MobileNetV1 graph shall be saved.

    """
    print('Reading the original MobileNetV1 model from {}...'.format(name_in))
    model = onnx.load(name_in)

    # Remove the redundant Pad node before the last AveragePool node (low-hanging fruit):
    model = optimizer.optimize(model, ['eliminate_nop_pad'])

    nodes = model.graph.node
    assert nodes[-1].op_type == 'Gemm', \
        'Expected a Gemm node as the last one in the graph.'
    _, gemm_weights_name, gemm_bias_name = nodes[-1].input

    # Remove the 3 nodes after the AveragePool node (with some guards)...
    expected_inputs_dict = dict(Constant=0, Reshape=2, Gemm=3)

    for removable_node in nodes[-3:]:
        expected_inputs = expected_inputs_dict[removable_node.op_type]
        actual_inputs = len(removable_node.input)
        assert actual_inputs == expected_inputs, \
            'Expected exactly {} input(s) for {} op node, got {}'.format(expected_inputs, \
                removable_node.op_type, actual_inputs)
        assert len(removable_node.output) == 1, 'Expected exactly one output for {} ' \
                'op node.'.format(removable_node.op_type)
        print('Removing {} node with inputs {} and outputs {} '
              'from the graph...'.format(removable_node.op_type,
                                         removable_node.input,
                                         removable_node.output))
        model.graph.node.remove(removable_node)

    # Store the output name of the AvgPool node (now the last node in the graph):
    avgpool_output_name = model.graph.node[-1].output[0]

    # Extend the weight dimensions of the original Gemm parameters with 2 trailing ones,
    # to make them compatible with a 1x1 Conv node...
    assert model.graph.initializer[-2].name == gemm_weights_name
    model.graph.initializer[-2].dims.extend([1, 1])

    # ... which we add to the graph here:
    last_conv_name = 'last_conv_fc'
    last_conv_inputs = [avgpool_output_name, gemm_weights_name, gemm_bias_name]
    model_output = model.graph.output[0]
    last_conv_outputs = [model_output.name]

    last_conv_fc = onnx.helper.make_node('Conv',
                                         name=last_conv_name,
                                         inputs=last_conv_inputs,
                                         outputs=last_conv_outputs,
                                         kernel_shape=[1, 1],
                                         strides=[1, 1],
                                         pads=[0, 0, 0, 0])
    print('Adding 1x1 Conv node {} with inputs {} and outputs {} '
          'to the graph...'.format(last_conv_name, last_conv_inputs,
                                   last_conv_outputs))
    model.graph.node.extend([last_conv_fc])

    # Make sure there are no dead ends after our graph surgery (maybe not necessary) and save it:
    # model = optimizer.optimize(model, ['eliminate_deadend'])
    print('Saving modified MobileNetV1 model to {}...'.format(name_out))
    onnx.save(model, name_out)
    print('Done.')


def main():
    """Look for the specific input file mobilenet_sym_no_bn.onnx, see README.md for details."""
    top = './Quantized MobileNet'
    if not path.isdir(top):
        raise RuntimeError('Could not find directory "{}". '
                           'Please read the README.md file.'.format(top))

    onnx_file_in = path.join(top, 'mobilenet_sym_no_bn.onnx')

    if not path.isfile(onnx_file_in):
        raise RuntimeError('Found directory {} but could not find "{}". '
                           'Please read the README.md file.'.format(
                               top, onnx_file_in))

    onnx_file_out = 'mobilenet_quantized_opt.onnx'
    optimize_quantized_mobilenet(onnx_file_in, onnx_file_out)


if __name__ == '__main__':
    main()
