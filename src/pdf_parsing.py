import os
import time
import logging
import re
import json
from tabulate import tabulate
from pathlib import Path
from typing import Iterable, List

# from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
# from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConversionResult

_log = logging.getLogger(__name__)

def _process_chunk(pdf_paths, pdf_backend, output_dir, num_threads, metadata_lookup, debug_data_path):
    """Helper function to process a chunk of PDFs in a separate process."""
    # Create a new parser instance for this process
    parser = PDFParser(
        pdf_backend=pdf_backend,
        output_dir=output_dir,
        num_threads=num_threads,
        csv_metadata_path=None  # Metadata lookup is passed directly
    )
    parser.metadata_lookup = metadata_lookup
    parser.debug_data_path = debug_data_path
    parser.parse_and_export(pdf_paths)
    return f"Processed {len(pdf_paths)} PDFs."

class PDFParser:
    def __init__(
        self,
        pdf_backend=DoclingParseV2DocumentBackend,
        output_dir: Path = Path("./parsed_pdfs"),
        num_threads: int = None,
        csv_metadata_path: Path = None,
    ):
        self.pdf_backend = pdf_backend
        self.output_dir = output_dir
        self.doc_converter = self._create_document_converter()
        self.num_threads = num_threads
        self.metadata_lookup = {}
        self.debug_data_path = None

        if csv_metadata_path is not None:
            self.metadata_lookup = self._parse_csv_metadata(csv_metadata_path)
            
        if self.num_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(self.num_threads)

    @staticmethod
    def _parse_csv_metadata(csv_path: Path) -> dict:
        """Parse CSV file and create a lookup dictionary with sha1 as key."""
        import csv
        metadata_lookup = {}
        
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Handle both old and new CSV formats for company name
                company_name = row.get('company_name', row.get('name', '')).strip('"')
                metadata_lookup[row['sha1']] = {
                    'company_name': company_name
                }
        return metadata_lookup

    def _create_document_converter(self) -> "DocumentConverter": # type: ignore
        """Creates and returns a DocumentConverter with default pipeline options."""
        from docling.document_converter import DocumentConverter, FormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, EasyOcrOptions
        from docling.datamodel.base_models import InputFormat
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
        
        pipeline_options = PdfPipelineOptions()
        # 光学，这意味着即使是扫描版 PDF（图片格式），也能把字读出来。
        pipeline_options.do_ocr = True
        # 只识别英文
        ocr_options = EasyOcrOptions(lang=['en'], force_full_page_ocr=False)
        pipeline_options.ocr_options = ocr_options
        # 表格识别
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        
        format_options = {
            InputFormat.PDF: FormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=pipeline_options,
                backend=self.pdf_backend
            )
        }
        
        return DocumentConverter(format_options=format_options)

    def convert_documents(self, input_doc_paths: List[Path]) -> Iterable[ConversionResult]:
        conv_results = self.doc_converter.convert_all(source=input_doc_paths)
        return conv_results
    
    def process_documents(self, conv_results: Iterable[ConversionResult]):
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        success_count = 0
        failure_count = 0

        for conv_res in conv_results:
            # 只遍历转换成的
            if conv_res.status == ConversionStatus.SUCCESS:
                success_count += 1
                # 初始化我们自定义的数据处理器
                processor = JsonReportProcessor(metadata_lookup=self.metadata_lookup, debug_data_path=self.debug_data_path)
                
                # Normalize the document data to ensure sequential pages
                # 导出Docling的原始字典
                data = conv_res.document.export_to_dict()
                # 标准化页码序列，防止 PDF 解析出来页码是 [1, 3, 4] 这种断层的，补全空页
                normalized_data = self._normalize_page_sequence(data)
                # 调用 processor 把数据重组为我们想要的格式
                processed_report = processor.assemble_report(conv_res, normalized_data)
                # 获取不带后缀的文件名
                doc_filename = conv_res.input.file.stem
                # 保存为json
                if self.output_dir is not None:
                    with (self.output_dir / f"{doc_filename}.json").open("w", encoding="utf-8") as fp:
                        json.dump(processed_report, fp, indent=2, ensure_ascii=False)
            else:
                failure_count += 1
                _log.info(f"Document {conv_res.input.file} failed to convert.")

        _log.info(f"Processed {success_count + failure_count} docs, of which {failure_count} failed")
        return success_count, failure_count

    def _normalize_page_sequence(self, data: dict) -> dict:
        """Ensure that page numbers in content are sequential by filling gaps with empty pages."""
        if 'content' not in data:
            return data
        
        # Create a copy of the data to modify
        normalized_data = data.copy()
        
        # Get existing page numbers and find max page
        existing_pages = {page['page'] for page in data['content']}
        max_page = max(existing_pages)
        
        # Create template for empty page
        # 创建空白页的模版
        empty_page_template = {
            "content": [],
            "page_dimensions": {}  # or some default dimensions if needed
        }
        
        # Create new content array with all pages
        new_content = []
        for page_num in range(1, max_page + 1):
            # Find existing page or create empty one
            #  如果这一页存在，就取这一页；如果不存在（StopIteration），就返回默认值（空页模板）
            page_content = next(
                (page for page in data['content'] if page['page'] == page_num),
                {"page": page_num, **empty_page_template}
            )
            new_content.append(page_content)
        
        normalized_data['content'] = new_content
        return normalized_data

    def parse_and_export(self, input_doc_paths: List[Path] = None, doc_dir: Path = None):
        start_time = time.time()

        if input_doc_paths is None and doc_dir is not None:
            input_doc_paths = list(doc_dir.glob("*.pdf"))
        
        total_docs = len(input_doc_paths)
        _log.info(f"Starting to process {total_docs} documents")
        # 转换
        conv_results = self.convert_documents(input_doc_paths)
        # 处理并保存
        success_count, failure_count = self.process_documents(conv_results=conv_results)
        elapsed_time = time.time() - start_time
        # 如果有失败的 抛出异常
        if failure_count > 0:
            error_message = f"Failed converting {failure_count} out of {total_docs} documents."
            failed_docs = "Paths of failed docs:\n" + '\n'.join(str(path) for path in input_doc_paths)
            _log.error(error_message)
            _log.error(failed_docs)
            raise RuntimeError(error_message)

        _log.info(f"{'#'*50}\nCompleted in {elapsed_time:.2f} seconds. Successfully converted {success_count}/{total_docs} documents.\n{'#'*50}")

    def parse_and_export_parallel(
        self,
        input_doc_paths: List[Path] = None,
        doc_dir: Path = None,
        optimal_workers: int = 10,
        chunk_size: int = None
    ):
        """Parse PDF files in parallel using multiple processes.
        
        Args:
            input_doc_paths: List of paths to PDF files to process
            doc_dir: Directory containing PDF files (used if input_doc_paths is None)
            optimal_workers: Number of worker processes to use. If None, uses CPU count.
        """
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Get input paths if not provided
        if input_doc_paths is None and doc_dir is not None:
            input_doc_paths = list(doc_dir.glob("*.pdf"))

        total_pdfs = len(input_doc_paths)
        _log.info(f"Starting parallel processing of {total_pdfs} documents")
        
        cpu_count = multiprocessing.cpu_count()
        
        # Calculate optimal workers if not specified
        if optimal_workers is None:
            optimal_workers = min(cpu_count, total_pdfs)
        # 尽量平均分配
        if chunk_size is None:
            # Calculate chunk size (ensure at least 1)
            chunk_size = max(1, total_pdfs // optimal_workers)
        # 分配块
        # Split documents into chunks
        chunks = [
            input_doc_paths[i : i + chunk_size]
            for i in range(0, total_pdfs, chunk_size)
        ]

        start_time = time.time()
        processed_count = 0
        # 创建进程池
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
            # Schedule all tasks
            futures = [
                executor.submit(
                    _process_chunk, 
                    chunk,
                    self.pdf_backend,
                    self.output_dir,
                    self.num_threads,
                    self.metadata_lookup,
                    self.debug_data_path
                )
                for chunk in chunks
            ]
            
            # Wait for completion and log results
            for future in as_completed(futures):
                try:
                    # 获取_process_chunk的结果
                    result = future.result()
                    # 解析返回字符串中的数据（比如 "Processed 2 PDFs"）来更新进度
                    processed_count += int(result.split()[1])  # Extract number from "Processed X PDFs"
                    _log.info(f"{'#'*50}\n{result} ({processed_count}/{total_pdfs} total)\n{'#'*50}")
                except Exception as e:
                    _log.error(f"Error processing chunk: {str(e)}")
                    raise

        elapsed_time = time.time() - start_time
        _log.info(f"Parallel processing completed in {elapsed_time:.2f} seconds.")


class JsonReportProcessor:
    def __init__(self, metadata_lookup: dict = None, debug_data_path: Path = None):
        self.metadata_lookup = metadata_lookup or {}
        self.debug_data_path = debug_data_path

    def assemble_report(self, conv_result, normalized_data=None):
        """Assemble the report using either normalized data or raw conversion result."""
        # 使用页码修复的数据
        data = normalized_data if normalized_data is not None else conv_result.document.export_to_dict()
        assembled_report = {}
        # 1. 组装元信息 (页数、公司名等)
        assembled_report['metainfo'] = self.assemble_metainfo(data)
        # 2. 组装正文内容 (把引用展开，按页码归类)
        assembled_report['content'] = self.assemble_content(data)
        # 3. 组装表格 (生成 Markdown 和 HTML)
        assembled_report['tables'] = self.assemble_tables(conv_result.document.tables, data)
        # 4. 组装图片
        assembled_report['pictures'] = self.assemble_pictures(data)
        self.debug_data(data)
        return assembled_report
    
    def assemble_metainfo(self, data):
        metainfo = {}
        # 从文件名获取SHA1哈希值，去掉后缀
        sha1_name = data['origin']['filename'].rsplit('.', 1)[0]
        metainfo['sha1_name'] = sha1_name
        # 统计数量
        metainfo['pages_amount'] = len(data.get('pages', []))
        metainfo['text_blocks_amount'] = len(data.get('texts', []))
        metainfo['tables_amount'] = len(data.get('tables', []))
        metainfo['pictures_amount'] = len(data.get('pictures', []))
        metainfo['equations_amount'] = len(data.get('equations', []))
        metainfo['footnotes_amount'] = len([t for t in data.get('texts', []) if t.get('label') == 'footnote'])
        
        # Add CSV metadata if available
        # 注入公司名称，检索时须知公司名称，提高性能
        if self.metadata_lookup and sha1_name in self.metadata_lookup:
            csv_meta = self.metadata_lookup[sha1_name]
            metainfo['company_name'] = csv_meta['company_name']
            
        return metainfo

    def process_table(self, table_data):
        # Implement your table processing logic here
        return 'processed_table_content'

    def debug_data(self, data):
        if self.debug_data_path is None:
            return
        doc_name = data['name']
        path = self.debug_data_path / f"{doc_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)    
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def expand_groups(self, body_children, groups):
        expanded_children = []

        for item in body_children:
            # 检查是否是一个引用 ($ref)，且引用的是 groups
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)

                if ref_type == 'groups':
                    group = groups[ref_num]
                    group_id = ref_num
                    group_name = group.get('name', '')
                    group_label = group.get('label', '')

                    for child in group['children']:
                        child_copy = child.copy()
                        child_copy['group_id'] = group_id
                        child_copy['group_name'] = group_name
                        child_copy['group_label'] = group_label
                        expanded_children.append(child_copy)
                else:
                    # 如果引用的不是 group（比如直接引用 text），就直接放进去
                    expanded_children.append(item)
            else:
                expanded_children.append(item)

        return expanded_children
    
    def _process_text_reference(self, ref_num, data):
        """Helper method to process text references and create content items.
        
        Args:
            ref_num (int): Reference number for the text item
            data (dict): Document data dictionary
            
        Returns:
            dict: Processed content item with text information
        """
        text_item = data['texts'][ref_num]
        item_type = text_item['label']
        content_item = {
            'text': text_item.get('text', ''),
            'type': item_type,
            'text_id': ref_num
        }
        
        # Add 'orig' field only if it differs from 'text'
        orig_content = text_item.get('orig', '')
        if orig_content != text_item.get('text', ''):
            content_item['orig'] = orig_content

        # Add additional fields if they exist
        if 'enumerated' in text_item:
            content_item['enumerated'] = text_item['enumerated']
        if 'marker' in text_item:
            content_item['marker'] = text_item['marker']
            
        return content_item
    
    def assemble_content(self, data):
        pages = {}
        # Expand body children to include group references
        # 获取文档主体的孩子节点
        body_children = data['body']['children']
        groups = data.get('groups', [])
        #  先把分组展开[文本1, 组A(文本2, 文本3)] -> 变成 -> [文本1, 文本2, 文本3]
        expanded_body_children = self.expand_groups(body_children, groups)

        # Process body content
        # 遍历每个内容块
        for item in expanded_body_children:
            # 必须是引用类型
            # 引用通常长这样：{ "$ref": "#/texts/123" }
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                # 获取类型和索引
                #  split('/') 把 "#/texts/123" 切割成 ["#", "texts", "123"]
                # [-2:] 取最后两个，即 ["texts", "123"]
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)

                # 如果这是个文本块
                if ref_type == 'texts':
                    text_item = data['texts'][ref_num]
                    # 调用辅助函数提取文本内容（比如清洗一下文字）
                    content_item = self._process_text_reference(ref_num, data)

                    # Add group information if available
                    # 如果之前是从 group 里展开的，把 group 信息带上
                    if 'group_id' in item:
                        content_item['group_id'] = item['group_id']
                        content_item['group_name'] = item['group_name']
                        content_item['group_label'] = item['group_label']
                    # 找到这个文本块所在的页码
                    # 'prov' 是 Provenance 的缩写，意思是“来源信息”
                    # Get page number from prov
                    if 'prov' in text_item and text_item['prov']:
                        page_num = text_item['prov'][0]['page_no']

                        # Initialize page if not exists
                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': text_item['prov'][0].get('bbox', {})
                            }
                        # 把整理好的文本扔进这一页的箱子里
                        pages[page_num]['content'].append(content_item)
                # table 与图片的处理方式一致
                elif ref_type == 'tables':
                    table_item = data['tables'][ref_num]
                    content_item = {
                        'type': 'table',
                        'table_id': ref_num
                    }

                    if 'prov' in table_item and table_item['prov']:
                        page_num = table_item['prov'][0]['page_no']

                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': table_item['prov'][0].get('bbox', {})
                            }

                        pages[page_num]['content'].append(content_item)
                
                elif ref_type == 'pictures':
                    picture_item = data['pictures'][ref_num]
                    content_item = {
                        'type': 'picture',
                        'picture_id': ref_num
                    }
                    
                    if 'prov' in picture_item and picture_item['prov']:
                        page_num = picture_item['prov'][0]['page_no']

                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': picture_item['prov'][0].get('bbox', {})
                            }
                        
                        pages[page_num]['content'].append(content_item)

        # 按照页码 (key) 进行排序，然后生成列表
        sorted_pages = [pages[page_num] for page_num in sorted(pages.keys())]
        return sorted_pages

    def assemble_tables(self, tables, data):
        assembled_tables = []
        # 遍历docling返回的tables列表
        for i, table in enumerate(tables):
            # 将 Docling 的 Table 对象转为标准的 Python 字典
            table_json_obj = table.model_dump()
            # 将表格数据转为 Markdown 字符串（LLM 最爱这种格式）
            table_md = self._table_to_md(table_json_obj)
            # 将表格转为 HTML 字符串（HTML 能表达合并单元格等复杂结构，比 Markdown 更强）
            table_html = table.export_to_html()
            

            # 从原始的data字典中拿到对应的表格数据
            table_data = data['tables'][i]
            # 提取页码 (prov = provenance 来源)
            table_page_num = table_data['prov'][0]['page_no']
            # 提取边界框 (Bounding Box)，即表格在页面上的坐标 [x, y, w, h] 这种格式
            table_bbox = table_data['prov'][0]['bbox']
            # 将 bbox 格式化为 [左, 上, 右, 下] 的列表，方便后续处理，相当于坐标
            table_bbox = [
                table_bbox['l'],
                table_bbox['t'], 
                table_bbox['r'],
                table_bbox['b']
            ]
            # 获取统计信息
            # Get rows and columns from the table data structure
            nrows = table_data['data']['num_rows']
            ncols = table_data['data']['num_cols']

            #  self_ref 类似 "#/tables/10"，这里通过 split 取出最后的数字 "10"
            ref_num = table_data['self_ref'].split('/')[-1]
            ref_num = int(ref_num)

            table_obj = {
                'table_id': ref_num,
                'page': table_page_num,
                'bbox': table_bbox,
                '#-rows': nrows,
                '#-cols': ncols,
                'markdown': table_md,
                'html': table_html,
                'json': table_json_obj
            }
            assembled_tables.append(table_obj)
        return assembled_tables

    def _table_to_md(self, table):
        # 专门把表格的 JSON 数据（Grid 结构）“打印”成 Markdown 文本。
        # Extract text from grid cells
        table_data = []
        for row in table['data']['grid']:
            table_row = [cell['text'] for cell in row]
            table_data.append(table_row)

        # 2. 生成 Markdown
        # Check if the table has headers
        if len(table_data) > 1 and len(table_data[0]) > 0:
            try:
                # headers=table_data[0]: 第一行作为表头
                # tablefmt="github": 生成 GitHub 风格的 Markdown 表格
                md_table = tabulate(
                    table_data[1:], headers=table_data[0], tablefmt="github"
                )
            except ValueError:
                md_table = tabulate(
                    table_data[1:],
                    headers=table_data[0],
                    tablefmt="github",
                    disable_numparse=True,
                )
        else:
            md_table = tabulate(table_data, tablefmt="github")
        
        return md_table

    def assemble_pictures(self, data):
        assembled_pictures = []
        for i, picture in enumerate(data['pictures']):
            # 处理图片内容
            # 图片可能有“孩子”节点（比如 OCR 识别出的图片内部文字）
            # 调用辅助函数 _process_picture_block 提取这些文字
            children_list = self._process_picture_block(picture, data)
            # 提取引用id
            ref_num = picture['self_ref'].split('/')[-1]
            ref_num = int(ref_num)
            
            # 提取位置
            picture_page_num = picture['prov'][0]['page_no']
            picture_bbox = picture['prov'][0]['bbox']
            picture_bbox = [
                picture_bbox['l'],
                picture_bbox['t'], 
                picture_bbox['r'],
                picture_bbox['b']
            ]
            
            picture_obj = {
                'picture_id': ref_num,
                'page': picture_page_num,
                'bbox': picture_bbox,
                'children': children_list,
            }
            assembled_pictures.append(picture_obj)
        return assembled_pictures
    
    def _process_picture_block(self, picture, data):
        children_list = []
        # # 遍历图片的 children (存放引用的列表)
        for item in picture['children']:
            # 检查是否是合法的引用格式
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)
                # 如果图片引用的是文本 (texts)
                # 这通常意味着 Docling 对图片做了 OCR，识别出了里面的字
                # 或者这是图片的标题 (Caption)
                if ref_type == 'texts':
                    content_item = self._process_text_reference(ref_num, data)
                        
                    children_list.append(content_item)

        return children_list
