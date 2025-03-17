datasets = [
    "Apache",
    "BGL",
    "Hadoop",
    "HDFS",
    "HealthApp",
    "HPC",
    "Linux",
    "Mac",
    "OpenSSH",
    "OpenStack",
    "Proxifier",
    "Spark",
    "Thunderbird",
    "Zookeeper"
]

filter_word = {
    'Apache': {
        'k': 3,
        'words': ["jni", "onShutdown", "vm"],
        'regs': []
    },
    'BGL': {
        'k': 4,
        'words': ["ciod", "Error", "error", "cr", "L3", "detected", "corrected", "ddr", "node"],
        'regs': [r"R\d{2}(?:-[A-Z0-9]{1,3})*", 
                 r"[a-fA-Z0-9]{8}",
                 r"[a-fA-F0-9]{20,}"]
    },
    'Hadoop': {
        'k': 3,
        'words': ["id"],
        'regs': [r"[a-zA-Z0-9-]+(_[a-z0-9-]+)+", 
                 r"[a-fA-F0-9]{7,8}",
                 r"\w+-(\d+)?"]
    },
    'HDFS': {
        'k': 3,
        'words': ["writeBlock", "for"],
        'regs': [r"blk_."]
        
    },
    'HealthApp': {
        'k': 3,
        'words': ["by", "open", "data"],
        'regs': [r"[a-f0-9]{7}", r"\d{4}-\d{2}-\d{2}"]
    },
    'HPC': {
        'k': 3,
        'words': ["Error"],
        'regs': [r"node-\d+(\\)?", 
                 r"node-D\d+", 
                 r"node-D\[(\d+(\\\s?\d+)*|\d+-\d+)\]",
                 r"node-\[(\d+(\\\s?\d+)*|\d+-\d+(\s?\\?\s?\d+-\d+)*|\d+(\\)*)\]", 
                 r"\d+\\"]
    },
    'Linux': {
        'k': 2,
        'words': ["node","user"],
        'regs': []
    },
    'Mac': {
        'k': 4,
        'words': ["lvl", "by","GoogleSoftwareUpdateAgent", "en0", "Off", "QQ"],
        'regs': [r"\d{4}-\d{2}-\d{2}", 
                 r"\d{2}:\d{2}:\d{2}\.\d{3,6}"]
    },
    'OpenSSH': {
        'k': 3,
        'words': [],
        'regs': []
    },
    'OpenStack': {
        'k': 3,
        'words': ["instance", "HTTP", "status", "len", "VM"],
        'regs': [r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}"]
    },
    'Proxifier': {
        'k': 6,
        'words': ["proxy", "bytes", "error", "sec"],
        'regs': [r'[\w]+-+']
    },
    'Spark': {
        'k': 2,
        'words': [],
        'regs': [r"attempt[\w]+",
                 r"[\w]+-[\w]+-[\d]+"]
    },
    'Thunderbird': {
        'k': 3,
        'words': ["root", "for", "user", "by", "with", "from"],
        'regs': [r"\d+-\d+", 
                 r"eth\d+"]
    },
    'Zookeeper': {
        'k': 3,
        'words': [],
        'regs': []
    }
}