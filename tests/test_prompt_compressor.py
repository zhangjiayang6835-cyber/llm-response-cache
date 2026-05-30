"""
PromptCompressor 测试
"""

import pytest
from llm_cache import PromptCompressor, CompressionLevel


class TestPromptCompressor:
    def test_compress_short_text_unchanged(self):
        """短文本（<50 tokens）跳过压缩"""
        pc = PromptCompressor()
        result = pc.compress("hi")
        assert result.compressed == "hi"
        assert result.compression_ratio == 0.0

    def test_mild_removes_fillers(self):
        """轻度压缩：去除废话"""
        pc = PromptCompressor(level=CompressionLevel.MILD)
        text = "好的，首先我们需要安装依赖，然后配置环境变量。"
        result = pc.compress(text)
        assert "好的" not in result.compressed
        assert "首先" not in result.compressed

    def test_standard_compresses_more(self):
        """标准压缩：去除过渡句"""
        pc = PromptCompressor(level=CompressionLevel.STANDARD)
        text = "换句话来说，基于上述分析我们可以得出这个结论。所谓的闪电贷攻击是指一种特殊的攻击方式。"
        result = pc.compress(text)
        assert "换句话来说" not in result.compressed
        assert "基于上述" not in result.compressed

    def test_aggressive_extreme_compression(self):
        """极省压缩"""
        pc = PromptCompressor(level=CompressionLevel.AGGRESSIVE)
        text = "我想要请你帮我写一个Solidity合约，这个合约需要实现ERC20标准的功能，可以吗？"
        result = pc.compress(text)
        # 极省会移除"想要""请你"
        assert result.compressed != text
        assert result.compression_ratio > 0.1

    def test_code_preserved(self):
        """代码块原样保留"""
        pc = PromptCompressor(level=CompressionLevel.AGGRESSIVE)
        text = """请写一个 Solidity 函数。代码如下：
```solidity
function transfer(address to, uint256 amount) public returns (bool) {
    _transfer(msg.sender, to, amount);
    return true;
}
```
谢谢！"""
        result = pc.compress(text)
        assert "```solidity" in result.compressed
        assert "function transfer" in result.compressed
        assert "msg.sender" in result.compressed

    def test_inline_code_preserved(self):
        """行内代码保留"""
        pc = PromptCompressor(level=CompressionLevel.AGGRESSIVE)
        text = "请解释 `delegatecall` 和 `call` 的区别"
        result = pc.compress(text)
        assert "`delegatecall`" in result.compressed
        assert "`call`" in result.compressed

    def test_ethereum_address_preserved(self):
        """以太坊地址保留"""
        pc = PromptCompressor(level=CompressionLevel.AGGRESSIVE)
        addr = "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb0"
        text = f"查询地址 {addr} 的余额"
        result = pc.compress(text)
        assert addr in result.compressed

    def test_url_preserved(self):
        """URL 保留"""
        pc = PromptCompressor(level=CompressionLevel.AGGRESSIVE)
        text = "参考文档 https://soliditylang.org/ 了解更多"
        result = pc.compress(text)
        assert "https://soliditylang.org/" in result.compressed

    def test_compression_ratio_positive(self):
        """长文本应该能压缩"""
        pc = PromptCompressor(level=CompressionLevel.STANDARD)
        text = (
            "首先，我想请你帮我写一个Python脚本。这个脚本的功能是读取CSV文件中的数据，"
            "然后按照日期进行分组统计，最后生成一个可视化的图表并保存为PNG格式。"
            "数据格式是第一列是日期，第二列是数值，第三列是分类标签。"
            "好的，如果你有任何问题或者需要更多信息，欢迎随时提出，谢谢！"
        )
        result = pc.compress(text)
        assert result.compression_ratio > 0.1
        assert result.saved_tokens > 0

    def test_stats_tracking(self):
        """统计追踪"""
        pc = PromptCompressor()
        pc.compress("这是一个很长的需要被压缩的测试文本，包含了很多废话和冗余信息。")
        pc.compress("另一个需要被压缩的很长的测试文本。")
        s = pc.stats()
        assert s["total_compressed"] == 2
        assert s["total_saved_tokens"] > 0
        assert s["avg_compression_ratio"] > 0

    def test_report(self):
        """报告输出"""
        pc = PromptCompressor()
        pc.compress("测试压缩报告生成功能。这是一段需要被压缩的测试文本。")
        report = pc.report()
        assert "压缩报告" in report
        assert "节省" in report

    def test_level_enum_acceptance(self):
        """接受字符串作为压缩级别"""
        pc = PromptCompressor(level="aggressive")
        assert pc.level == CompressionLevel.AGGRESSIVE

    def test_no_negative_compression(self):
        """压缩后不会比原来长"""
        pc = PromptCompressor(level=CompressionLevel.AGGRESSIVE)
        text = "好的，这是一个非常简单的测试文本，需要被压缩处理。"
        result = pc.compress(text)
        assert result.compressed_tokens <= result.original_tokens
