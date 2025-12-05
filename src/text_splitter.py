import json
import tiktoken
from pathlib import Path
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextSplitter():
    def _get_serialized_tables_by_page(self, tables: List[Dict]) -> Dict[int, List[Dict]]:
        """Group serialized tables by page number"""
        # 把一个杂乱的表格列表，整理成一个按页码索引的字典。
        # 结构将会是：{ 页码1: [表格A, 表格B], 页码2: [表格C] }
        tables_by_page = {}
        for table in tables:
            if 'serialized' not in table:
                continue
            # 只有不被serialized的表格才会被处理
            page = table['page']
            # 如果字典里还没这个页码的坑位，先挖个坑（初始化空列表）
            if page not in tables_by_page:
                tables_by_page[page] = []

            # 提取并合并描述文本
            # table["serialized"]["information_blocks"] 是一个列表，里面是一句句 AI 写的话。
            # 这里用 "\n".join 把它们拼成一段完整的自然语言描述。
            table_text = "\n".join(
                block["information_block"] 
                for block in table["serialized"]["information_blocks"]
            )
            # 4. 组装数据并存入字典
            tables_by_page[page].append({
                "page": page,
                "text": table_text,
                "table_id": table["table_id"],
                "length_tokens": self.count_tokens(table_text)
            })
            
        return tables_by_page

    def _split_report(self, file_content: Dict[str, any], serialized_tables_report_path: Optional[Path] = None) -> Dict[str, any]:
        """Split report into chunks, preserving markdown tables in content and optionally including serialized tables."""
        # 将报告切分为 chunks，保留内容中的 Markdown 表格，并可选地包含序列化表格。
        chunks = []
        chunk_id = 0
        
        tables_by_page = {}
        # 读取原始的 parsed report (因为 merged report 可能丢掉了表格的详细序列化信息)
        if serialized_tables_report_path is not None:
            with open(serialized_tables_report_path, 'r', encoding='utf-8') as f:
                parsed_report = json.load(f)
            # 调用上面那个函数，把表格整理成 {页码: 描述} 的格式
            tables_by_page = self._get_serialized_tables_by_page(parsed_report.get('tables', []))
        # 按页遍历并切分
        # file_content['content']['pages'] 是 merged 后的正文数据
        for page in file_content['content']['pages']:
            # 调用 _split_page 把这一页的 Markdown 文本切成多个 300 token 的小块
            page_chunks = self._split_page(page)
            # 给每个切块打上标记
            for chunk in page_chunks:
                chunk['id'] = chunk_id
                chunk['type'] = 'content'
                chunk_id += 1
                chunks.append(chunk)
            # 给每个表格打上标记
            # 检查当前页码是否有对应的表格描述
            if tables_by_page and page['page'] in tables_by_page:
                for table in tables_by_page[page['page']]:
                    table['id'] = chunk_id
                    table['type'] = 'serialized_table'
                    chunk_id += 1
                    chunks.append(table)
        # 将生成的 chunks 列表挂载到 content 下面
        file_content['content']['chunks'] = chunks
        return file_content

    def count_tokens(self, string: str, encoding_name="o200k_base"):
        # 计算一段文本包含多少个 Token。
        encoding = tiktoken.get_encoding(encoding_name)

        tokens = encoding.encode(string)
        token_count = len(tokens)

        return token_count

    def _split_page(self, page: Dict[str, any], chunk_size: int = 300, chunk_overlap: int = 50) -> List[Dict[str, any]]:
        """Split page text into chunks. The original text includes markdown tables."""
        # 将页面文本切分成块。原始文本包含 Markdown 表格
        # 调用langchain的RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        # 切分这一页
        chunks = text_splitter.split_text(page['text'])
        # 切好之后包装元数据和页码信息
        chunks_with_meta = []
        for chunk in chunks:
            chunks_with_meta.append({
                "page": page['page'],
                "length_tokens": self.count_tokens(chunk),
                "text": chunk
            })
        return chunks_with_meta

    def split_all_reports(self, all_report_dir: Path, output_dir: Path, serialized_tables_dir: Optional[Path] = None):
        # 获取所有待处理的 JSON,处理为切块之后存入新的文件夹
        all_report_paths = list(all_report_dir.glob("*.json"))
        
        for report_path in all_report_paths:
            # 确定原始数据路径（用于提取表格描述）
            serialized_tables_path = None
            if serialized_tables_dir is not None:
                serialized_tables_path = serialized_tables_dir / report_path.name
                if not serialized_tables_path.exists():
                    print(f"Warning: Could not find serialized tables report for {report_path.name}")
            # 读取 Merged JSON
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
                
            updated_report = self._split_report(report_data, serialized_tables_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / report_path.name, 'w', encoding='utf-8') as file:
                json.dump(updated_report, file, indent=2, ensure_ascii=False)
                
        print(f"Split {len(all_report_paths)} files")
