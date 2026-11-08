"""Evaluation task definitions.

Each minishop task copies the correct codebase (eval/base_repos/minishop) and injects one or
more bugs. The correct codebase doubles as the reference solution, and the hidden tests in
eval/base_repos/minishop/hidden_tests are only used when a run is judged.
"""

MINISHOP_TASKS = [
    {
        "id": "ms01_discount_before_tax",
        "category": "逻辑错误",
        "task": "金牌（gold）会员订单的税额算错了：税应该按扣除折扣之后的金额计算，现在像是按折扣前的金额在算。请修复价格计算，并保证所有测试通过。",
        "injections": [
            ("minishop/pricing.py", "        taxable = line.subtotal - discount * share\n", "        taxable = line.subtotal\n"),
        ],
    },
    {
        "id": "ms02_parser_regex",
        "category": "正则与转义",
        "task": "订单文本解析无法识别 SKU 中带连字符的行（例如 `TEA-01 x2`），也不接受 x 和数量之间有空格的写法（例如 `COF-10 x 1`）。请修复，让这两种写法都能正确解析，其他行为保持不变。",
        "injections": [
            (
                "minishop/parser.py",
                r'LINE_RE = re.compile(r"^(?P<sku>[A-Z0-9][A-Z0-9-]*)\s+x\s*(?P<qty>\d+)$", re.IGNORECASE)',
                r'LINE_RE = re.compile(r"^(?P<sku>[A-Z0-9]+)\s+x(?P<qty>\d+)$", re.IGNORECASE)',
            ),
        ],
    },
    {
        "id": "ms03_invoice_format",
        "category": "字符串格式",
        "task": "发票（invoice）文本的格式不符合规范：商品名应该用双引号括起来，并且整段文本要以换行符结尾。请修复发票渲染。",
        "injections": [
            (
                "minishop/report.py",
                """            f'{line.quantity:>3} x "{line.product.name}" @ {format_money(line.product.unit_price)}'""",
                """            f"{line.quantity:>3} x '{line.product.name}' @ {format_money(line.product.unit_price)}\"""",
            ),
            ("minishop/report.py", r'    return "\n".join(lines) + "\n"', r'    return "\n".join(lines)'),
        ],
    },
    {
        "id": "ms04_inventory_two_bugs",
        "category": "多处修复",
        "task": "库存模块有两个问题：预留（reserve）之后可用库存没有相应减少；取消预留（release）之后，预留数量反而增加了。请修复。",
        "injections": [
            (
                "minishop/inventory.py",
                "        return self.on_hand(sku) - self.reserved(sku)\n",
                "        return self.on_hand(sku)\n",
            ),
            (
                "minishop/inventory.py",
                '            raise ValueError(f"{sku}: cannot release {quantity}, only {self.reserved(sku)} reserved")\n'
                "        self._reserved[sku] = self.reserved(sku) - quantity\n",
                '            raise ValueError(f"{sku}: cannot release {quantity}, only {self.reserved(sku)} reserved")\n'
                "        self._reserved[sku] = self.reserved(sku) + quantity\n",
            ),
        ],
    },
    {
        "id": "ms05_money_rounding",
        "category": "数值精度",
        "task": "金额的四舍五入不正确：像 2.675 这样的金额应该四舍五入到分得到 2.68，现在却得到 2.67。请修复金额处理，避免浮点数误差。",
        "injections": [
            (
                "minishop/money.py",
                "    return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)\n",
                "    return Decimal(round(float(value), 2)).quantize(CENT)\n",
            ),
        ],
    },
    {
        "id": "ms06_bulk_threshold",
        "category": "跨模块定位",
        "task": "测试发现：同一商品买满 10 件时，发票上没有体现批量折扣（满 10 件应享受 3% 的折扣）。请找到原因并修复。",
        "injections": [
            ("minishop/pricing.py", "        if line.quantity >= BULK_THRESHOLD:\n", "        if line.quantity > BULK_THRESHOLD:\n"),
        ],
    },
    {
        "id": "ms07_bogo_coupon",
        "category": "功能实现",
        "task": "实现 BOGO 优惠券：优惠码格式为 `BOGO-<SKU>`（不区分大小写），对订单中该 SKU 的商品买一送一，即每两件中有一件免费（奇数件时多出的一件照常收费）。折扣在计税之前生效。",
        "injections": [
            (
                "minishop/coupons.py",
                '    if code.startswith("BOGO-"):\n'
                '        sku = code[len("BOGO-"):]\n'
                "        for line in order.lines:\n"
                "            if line.product.sku.upper() == sku:\n"
                "                free_units = line.quantity // 2\n"
                "                return to_money(line.product.unit_price * free_units)\n"
                '        return Decimal("0.00")\n',
                '    if code.startswith("BOGO-"):\n'
                "        # TODO: buy-one-get-one coupons are not implemented yet.\n"
                '        return Decimal("0.00")\n',
            ),
        ],
    },
    {
        "id": "ms08_catalog_price_range",
        "category": "大文件定位",
        "task": "商品目录的价格区间查询 price_range(low, high) 应该同时包含上下两个边界，但查询 3.50 到 3.50 时，找不到价格正好是 3.50 的商品。请修复。",
        "injections": [
            (
                "minishop/catalog.py",
                "        (p for p in PRODUCTS.values() if low <= p.unit_price <= high),\n",
                "        (p for p in PRODUCTS.values() if low <= p.unit_price < high),\n",
            ),
        ],
    },
]

# Function tasks come from HumanEval. `reference` (the canonical solution) replaces the final
# `pass` during validation, so build_tasks.py can prove every task is solvable.
FUNCTION_TASKS = [
    {
        "id": "fn01_make_palindrome",
        "category": "函数实现",
        "source": "functions/HumanEval_10",
        "task": "实现 solution.py 中的 make_palindrome 函数，使其通过测试。",
        "reference": (
            "    if not string:\n"
            "        return ''\n"
            "    beginning_of_suffix = 0\n"
            "    while not is_palindrome(string[beginning_of_suffix:]):\n"
            "        beginning_of_suffix += 1\n"
            "    return string + string[:beginning_of_suffix][::-1]\n"
        ),
    },
    {
        "id": "fn02_remove_duplicates",
        "category": "函数实现",
        "source": "functions/HumanEval_26",
        "task": "实现 solution.py 中的 remove_duplicates 函数，使其通过测试。",
        "reference": (
            "    import collections\n"
            "    counts = collections.Counter(numbers)\n"
            "    return [n for n in numbers if counts[n] <= 1]\n"
        ),
    },
    {
        "id": "fn03_find_zero",
        "category": "函数实现",
        "source": "functions/HumanEval_32",
        "task": "实现 solution.py 中的 find_zero 函数，使其通过测试。",
        "reference": (
            "    begin, end = -1.0, 1.0\n"
            "    while poly(xs, begin) * poly(xs, end) > 0:\n"
            "        begin *= 2.0\n"
            "        end *= 2.0\n"
            "    while end - begin > 1e-10:\n"
            "        center = (begin + end) / 2.0\n"
            "        if poly(xs, center) * poly(xs, begin) > 0:\n"
            "            begin = center\n"
            "        else:\n"
            "            end = center\n"
            "    return begin\n"
        ),
    },
    {
        "id": "fn04_decode_cyclic",
        "category": "函数实现",
        "source": "functions/HumanEval_38",
        "task": "实现 solution.py 中的 decode_cyclic 函数，使其通过测试。",
        "reference": "    return encode_cyclic(encode_cyclic(s))\n",
    },
]
