#!/usr/bin/env python3
"""
Cross-platform compatibility tool: Convert multi-line shell commands with backslash (\) line continuation to single-line commands
Usage: python convert_to_single_line.py input_file [output_file]
If no output file is specified, the input file will be overwritten
"""

import sys
import os
import re

def convert_multiline_to_single_line(file_path, output_path=None):
    """
    Convert multi-line commands (ending with backslash) to single-line commands
    """
    if output_path is None:
        output_path = file_path

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Preserve the original file permissions
        original_mode = os.stat(file_path).st_mode
            
        # Use regular expression to find lines ending with \ and remove newlines and \
        pattern = r'\\\s*\n\s*'
        converted_content = re.sub(pattern, ' ', content)
            
        # Write the converted content to the output file
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(converted_content)
            
        # Restore the original file permissions
        os.chmod(output_path, original_mode)
        
        print(f"Successfully converted {file_path} to single-line command format, saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def convert_single_line_to_multiline(file_path, output_path=None, line_prefix="--"):
    """
    Convert single-line commands back to multi-line format (using \ continuation), with each parameter on a separate line
    Mainly used to split parameters starting with line_prefix into separate lines
    """
    if output_path is None:
        output_path = file_path

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Preserve the original file permissions
        original_mode = os.stat(file_path).st_mode
        
        # Find command lines and convert them
        lines = content.split('\n')
        converted_lines = []
        
        for line in lines:
            if line_prefix in line:
                # Split the command line
                parts = line.split(line_prefix)
                # Save the first part (command name)
                first_part = parts[0].rstrip() + " \\"
                converted_lines.append(first_part)
                
                # Process each parameter
                for i, part in enumerate(parts[1:], 1):
                    if part.strip():
                        # If not the last parameter, add continuation character
                        if i < len(parts) - 1:
                            converted_lines.append(line_prefix + part.rstrip() + " \\")
                        else:
                            converted_lines.append(line_prefix + part.rstrip())
            else:
                converted_lines.append(line)
        
        # Write the converted content to the output file
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(converted_lines))
        
        # Restore the original file permissions
        os.chmod(output_path, original_mode)
        
        print(f"Successfully converted {file_path} to multi-line command format, saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} input_file [output_file] [--to-multiline]")
        return
    
    file_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    
    # Check if conversion to multi-line is needed
    to_multiline = '--to-multiline' in sys.argv
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return
    
    if to_multiline:
        convert_single_line_to_multiline(file_path, output_path)
    else:
        convert_multiline_to_single_line(file_path, output_path)

if __name__ == "__main__":
    main()
