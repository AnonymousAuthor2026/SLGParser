import re
import pandas as pd
from LogProcessor import preprocess_log,process_insert_or_update_template


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_template = False
        self.template_id = None


class TrieTree:
    def __init__(self):
        self.root = TrieNode()
        self.current_id = 0
        self.tree = ""

    def insert(self, template: list):
        node = self.root
        try:
            for token in template:

                regex_token = self.generate_regex(token)
                compiled_token = re.compile(regex_token)
                
                match_found = False
                for key, child in node.children.items():
                    if key.pattern == compiled_token.pattern:
                        node = child
                        match_found = True
                        break
                
                if not match_found:
                    node.children[compiled_token] = TrieNode()
                    node = node.children[compiled_token]

            if node.template_id is None:
                node.template_id = self.current_id
                self.current_id += 1
            
            node.is_end_of_template = True
            return node.template_id

        except Exception as e:
            print(f"Exception raised: {e} ; and template is " + template[1])
            return -1
        except re.error as re_err:
            print(f"Exception raised: {e} ; and template is " + template[1])
            return -1
            
    
    def update(self, template, template_id):
        """
        Update the template in the trie tree.

        returns:
            template_id: the updated template ID
        """
        node = self.root

        regex_loglen_str = self.generate_regex(template[0])
        regex_loglen = re.compile(regex_loglen_str)
        
        regex_str = self.generate_regex(template[1])
        compiled_token = re.compile(regex_str)

        # search the first layer
        for length, child in node.children.items():
            if length.pattern == regex_loglen.pattern:
                # search the second layer
                for token, deep_child in child.children.items():
                    if deep_child.template_id == template_id:
                        # update node
                        existing_children = deep_child.children
                        # delete old node
                        del child.children[token]
                        # create new node
                        child.children[compiled_token] = TrieNode()
                        new_node = child.children[compiled_token]
                        # copy
                        new_node.template_id = template_id
                        new_node.is_end_of_template = True
                        new_node.children = existing_children
                        return new_node.template_id
        return None

    def build_tree(self, node=None, level=0):
        """
        Build the tree of the trie tree.
        """
        if node is None:
            node = self.root
        # current node info
        for token, child_node in node.children.items():
            indentation = "  " * level
            node_info = f"{token} (Id: {child_node.template_id}, End: {child_node.is_end_of_template})"
            self.tree += f"{indentation}{node_info}\n"
            # reverse child_node
            self.build_tree(child_node, level + 1)

    def get_tree(self):
        return self.tree

    def _match(self, node, log, index):
        if node.is_end_of_template:
            return (True, node.template_id)
        
        for token_regex, child in node.children.items():
            
            try:
                if token_regex.fullmatch(log[index]):
                    _match_result = self._match(child, log, index+1)
                    if _match_result[0]:
                        return _match_result
                    
            except IndexError:
                print(f"IndexError: log[{index}] is out of range")
            except Exception as e:
                print(f"Unexpected error: {e} ; log entry: {log[index] if index < len(log) else 'OUT_OF_RANGE'}")
            
        return (False, None)
    
    def _match_with_filterId(self, node, log, index, filter_ids):
        if node.is_end_of_template:
            if node.template_id not in filter_ids:
                return (True, node.template_id)
        
        for token_regex, child in node.children.items():
            
            try:
                if token_regex.fullmatch(log[index]):
                    _match_result = self._match_with_filterId(child, log, index+1, filter_ids)
                    if _match_result[0]:
                        return _match_result
            
            except IndexError:
                print(f"IndexError: log[{index}] is out of range")
            except Exception as e:
                print(f"Unexpected error: {e} ; log entry: {log[index] if index < len(log) else 'OUT_OF_RANGE'}")
            
        return (False, None)

    def search(self, log, filter_ids):
        return self._match(self.root, log, 0) if len(filter_ids) == 0 else self._match_with_filterId(self.root, log, 0, filter_ids)

    def generate_regex(self, token: str):
        # if "<*>" in token:
            # if "<*> <*> <*>" in pattern:
            #     pattern = re.sub(r"<\*> <\*> <\*>",r"<*>",pattern)
        while "<*> <*>" in token:
            token = re.sub(r"<\*>\s<\*>", "<*>", token)

        token = token.replace('\\', '\\\\')
        token = re.sub(r"(?<!<)\*(?!>)", r"\*", token)
        token = token.replace('[', r'\[')
        token = token.replace(']', r'\]')
        token = token.replace('(', r'\(')
        token = token.replace(')', r'\)')
        token = token.replace('-', r'\-')
        token = token.replace('+', r'\+')
        
        token = re.sub(r"\s*<\*>\s*", ".*", token)
            
        return f"^{token}$"


class LogMatcher:
    def __init__(self):
        self.trie = TrieTree()
        self.template_id = 0
        self.templates = []

    def insert(self, raw_log, template):
        length = len(raw_log.split())
        template_insert = process_insert_or_update_template(length, template)
        
        template_id = self.trie.insert(template_insert)
        if template_id is not None:
            self.templates.append(template)
        return template_id
    
    def update(self, raw_log, template, template_id):
        length = len(raw_log.split())
        template_update = process_insert_or_update_template(length, template)
        template_id_ret = self.trie.update(template_update, template_id)
        
        if template_id_ret is not None:
            self.templates[template_id_ret] = template
            return template_id_ret
        return template_id_ret

    def get_tree(self):
        self.trie.build_tree()
        return self.trie.get_tree()
    
    def match_template(self, log, filter_ids):
        """
        preprocess log, and the match one template in matcher

        returns:
            True or False: The flag of matching result. 
            template_id: The matched template ID.
        """
        preprocess_content = preprocess_log(log)
        return self.trie.search(preprocess_content, filter_ids)