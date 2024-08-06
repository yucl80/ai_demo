import javalang

def parse_java_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        java_code = file.read()
    
    tree = javalang.parse.parse(java_code)
    return tree

def extract_method_calls(tree):
    method_calls = []
    for path, node in tree.filter(javalang.tree.MethodInvocation):
        method_calls.append({
            'method_name': node.member,
            'arguments': [arg.name for arg in node.arguments] if node.arguments else []
        })
    return method_calls

def extract_lambda_method_calls(tree):
    lambda_method_calls = []
    for path, node in tree.filter(javalang.tree.LambdaExpression):
        for call_path, call_node in tree.filter(javalang.tree.MethodInvocation):
            print(call_node.arguments)
            lambda_method_calls.append({
                'lambda_expression': node,
                'method_name': call_node.member,
                # 'arguments': [arg.name for arg in call_node.arguments] if call_node.arguments else []
            })
    return lambda_method_calls

def extract_method_call_chains(tree):
    method_call_chains = {}
    for path, node in tree.filter(javalang.tree.MethodDeclaration):
        method_name = node.name
        method_call_chains[method_name] = []
        for call_path, call_node in tree.filter(javalang.tree.MethodInvocation, path):
            method_call_chains[method_name].append(call_node.member)
    return method_call_chains

if __name__ == "__main__":
    # java_file_path = 'path/to/your/JavaFile.java'
    # tree = parse_java_file(java_file_path)
    
    new_code = """
    public class OrderDao{
    private double queryOrder(Order order){
        List<Order> orderList = jdbcTemplate.query(order);
        orderList = orderList.stream().filter(o -> o.getOrderStatus() == OrderStatus.Paid ).collect(Collectors.toList());
        return order;
    }    
    }
    """
    
    tree = javalang.parse.parse(new_code)
    
    lambda_method_calls = extract_lambda_method_calls(tree)
    
    print("Lambda Method Calls:", lambda_method_calls)