import re
from typing import Tuple

def token_postprocess(message):
    new_message = []
    for token in message:
        while "<*><*>" in token:
            token = token.replace("<*><*>", '<*>')
        new_message.append(token)
    message_result = ' '.join(new_message)
    return message_result


def token_split(message):
    message_parts = re.split(r'\s+', message)
    message_parts = [message_part for message_part in message_parts if message_part]
    return message_parts


def preprocess_log(log):
    log_length = len(log.split())

    datetime_regexs = [r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$',
             r'[A-Za-z]{3} [A-Za-z]{3} \d{1,2} \d{2}:\d{2}:\d{2} [A-Z]{3} \d{4}$',
             r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \+\d{4}$']
    for datetime_regex in datetime_regexs:
        date_finds = re.findall(datetime_regex, log)
        if date_finds:
            for find in date_finds:
                new_find = re.sub(r' ', '', find)
                log = re.sub(re.escape(find), new_find, log)
    
    log = re.sub(r'[~`@#$%^&_=:;"\',.?/|{}]+', '', log)

    log_parts = token_split(log)
    log_result = ' '.join(log_parts).strip()
    return [str(log_length), log_result]


def process_insert_or_update_template(length, template):

    template = re.sub(r'[~`@#$%^&_=:;"\',.?/|{}]+', '', template)

    template_list = token_split(template)
    log_template = token_postprocess(template_list)

    return [str(length), log_template]


def postprocess_template(template):

    template = re.sub(r'\{(\w+)\}', '<*>',template)

    while bool(re.search(r'(/[\w-]+)+<\*>', template)):
        template = re.sub(r'(/[\w-]+)+<\*>', '<*>', template)

    template = correct_single_template(template)

    template = fix_template(template)

    template = re.sub(r'<\*>\s*(KB|GB|MB)', '<*>', template, flags=re.IGNORECASE)
    template = re.sub(r'node\-D?(\d+|<\*>|\[[D<\*>\-\s]+\])', '<*>', template)
    template = re.sub(r'(<\*>\s<\*>\s<ok>\s){2,}<\*>\s<\*>\s<ok>', '<*> <*> <ok> <*> <*> <ok>', template)
    template = re.sub(r'(?<=[\s=])/([\w<*>-]+/)*[\w<*>-]+', '<*>', template)
    template = re.sub(r'::ffff:<\*>', '<*>', template)
    template = re.sub(r'\[[\w]+\.\w+:<\*>\]', '[<*>]', template)

    if template.endswith("="):
        template = template + "<*>"

    # return template
    return correct_single_template(template)

def fix_template(template):
    log_template_list = re.split(r'([\s:=,()[\]#-]+|\.\.+)', template)
    log_template_list = [re.sub(r'^-?\d+(\.\d+)?$', '<*>', s) for s in log_template_list]
    template = "".join(log_template_list)

    template = re.sub(r'(\w+[\.\$@]){2,}<\*>', '<*>', template)
    template = re.sub(r'\( (<\*>\s*){3,} \)', '( <*> )', template)
    template = re.sub(r'(<\*>\s){4,}<\*>', '<*>', template)
    template = re.sub(r'(<\*>-){2,}<\*>', '<*>',template)
    
    return template


def correct_single_template(template):

    path_delimiters = {
        r'\s', r'\,', r'\!', r'\;', r'\:',
        r'\=', r'\|', r'\"', r'\'',
        r'\[', r'\]', r'\(', r'\)', r'\{', r'\}'
    }
    token_delimiters = path_delimiters.union({
        r'\.', r'\-', r'\+', r'\@', r'\#', r'\$', r'\%', r'\&',
    })

    template = template.strip()
    template = re.sub(r'\s+', ' ', template)

    tokens = re.split('(' + '|'.join(token_delimiters) + ')', template)
    new_tokens = []
    for token in tokens:
        if re.match(r'^[^\s\/]*<\*>[^\s\/]*$', token):
            if token != '<*>/<*>':
                token = '<*>'
        new_tokens.append(token)
    template = ''.join(new_tokens)
    while True:
        prev = template
        template = re.sub(r'<\*>\.<\*>', '<*>', template)
        if prev == template:
            break
    while True:
        prev = template
        template = re.sub(r'<\*><\*>', '<*>', template)
        if prev == template:
            break
    # New
    while "<*>##<*>" in template:
        template = template.replace("<*>##<*>", "<*>")

    while " #<*># " in template:
        template = template.replace(" #<*># ", " <*> ")

    while " #<*> " in template:
        template = template.replace(" #<*> ", " <*> ")

    while "<*>:<*>" in template:
        template = template.replace("<*>:<*>", "<*>")

    while "<*>#<*>" in template:
        template = template.replace("<*>#<*>", "<*>")
    
    while "#<*>#" in template:
        template = template.replace("#<*>#", "<*>")
    # New
    while "0x<*>" in template:
        template = template.replace("0x<*>", "<*>")

    while "<*>/<*>" in template:
        template = template.replace("<*>/<*>", "<*>")
    # New
    while " /<*>" in template and " //<*>" not in template:
        template = template.replace(" /<*>", " <*>")

    while "<*>@<*>" in template:
        template = template.replace("<*>@<*>", "<*>")

    while "<*>.<*>" in template:
        template = template.replace("<*>.<*>", "<*>")
    return template


def contain_wildcard_only(template):
    template = template.strip()
    template = re.sub(r'<\*>', '', template)
    template = re.sub(r'[^\w-]', '', template)
    if template == '' or template == '<*>':
        return True
    return False


def postprocess_error_template(log, 
                               template_id, 
                               matcher, 
                               filter_ids) -> Tuple[str, int]:
    regs_common = []
    patterns = [
        "((?<=[^A-Za-z0-9])|^)(([0-9a-f]{2,}:){3,}([0-9a-f]{2,}))((?=[^A-Za-z0-9])|$)",
        "((?<=[^A-Za-z0-9])|^)(\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3})((?=[^A-Za-z0-9])|$)",
        "((?<=[^A-Za-z0-9])|^)([0-9a-f]{6,} ?){3,}((?=[^A-Za-z0-9])|$)",
        "((?<=[^A-Za-z0-9])|^)([0-9A-F]{4} ?){4,}((?=[^A-Za-z0-9])|$)",
        "((?<=[^A-Za-z0-9])|^)(0x[a-f0-9A-F]+)((?=[^A-Za-z0-9])|$)",
        "((?<=[^A-Za-z0-9])|^)(<[a-f0-9A-F]+>)((?=[^A-Za-z0-9])|$)",
        "((?<=[^A-Za-z0-9])|^)([\\-\\+]?\\d+)((?=[^A-Za-z0-9])|$)",
        "(?<=executed cmd )(\".+?\")"
        ]
    for pattern in patterns:
        regs_common.append(re.compile(pattern))
    template = log
    for reg in regs_common:
        template = reg.sub("<*>", template)

    template = postprocess_template(template)
    if template_id is not None:
        template_id = matcher.update(log, template, template_id)
    else:
        template_id = matcher.insert(log, template)
    match_result = matcher.match_template(log, filter_ids)
    if match_result[0]:
        if template_id == match_result[1]:
            print(f"Update successfully! The new template is: {template}")
            return template, template_id
    else:
        template_id = matcher.update(log, log, template_id) if template_id is not None else matcher.insert(log, log)
        
    print(f"Update successfully! The new template is: {log}")
    return log, template_id