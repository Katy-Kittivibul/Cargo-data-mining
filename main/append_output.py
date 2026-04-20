import codecs

with codecs.open('e:/Coding Projects/Cargo/main/terminal_output.txt', 'r', 'utf-16le') as f:
    text = f.read()

with open('e:/Coding Projects/Cargo/main/findings_summary.md', 'a', encoding='utf-8') as f:
    f.write('\n\n## 8. Full Terminal Output\n\n```text\n' + text + '\n```\n')
