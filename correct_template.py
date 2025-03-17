import re
import pandas as pd
from settings import datasets

"""
Apache
"""
dataset = "Apache"
Apache_df = pd.read_csv(f"full_dataset/{dataset}/{dataset}_full.log_structured.csv")
Apache_items_to_be_replaced = [
    "[client <*>] mod_security: Access denied with code <*>. Error reading POST data, error_code=<*> [hostname <*>] [uri <*>]"
    ]
Apache_new_items = [
    '[client <*>] mod_security: Access denied with code <*>. Error reading POST data, error_code=<*> [hostname "<*>"] [uri "<*>"]'
    ]
for old, new in zip(Apache_items_to_be_replaced,Apache_new_items):
    Apache_df.loc[Apache_df["EventTemplate"] == old, "EventTemplate"] = new
Apache_df.to_csv(f"full_dataset/{dataset}/{dataset}_full.log_structured_new.csv", index=False, encoding='utf-8')


"""
BGL
"""
dataset = "BGL"
BGL_df = pd.read_csv(f"full_dataset/{dataset}/{dataset}_full.log_structured.csv")
BGL_items_to_be_replaced = [
    "<*> ddr errors(s) detected and corrected on rank <*>, symbol <*> bit <*>",
    "d-cache flush parity error.......<*>",
    "ciod: for node <*> read continuation request but ioState is <*>",
    "BglIdoChip table has <*> IDOs with the same IP address <*>",
    "Node card VPD check: missing <*>, VPD ecid <*> in processor card slot <*>",
    "round toward zero...................<*>"
    ]
BGL_new_items = [
    "<*> ddr errors(s) detected and corrected on rank <*>, symbol <*>, bit <*>",
    "d-cache flush parity error........<*>",
    "ciod: for node <*>, read continuation request but ioState is <*>",
    "BglIdoChip table has <*> IDOs with the same IP address (<*>)",
    "Node card VPD check: missing <*> node, VPD ecid <*> in processor card slot <*>",
    "round toward zero........................<*>"
    ]
for old, new in zip(BGL_items_to_be_replaced,BGL_new_items):
    BGL_df.loc[BGL_df["EventTemplate"] == old, "EventTemplate"] = new
BGL_df.to_csv(f"full_dataset/{dataset}/{dataset}_full.log_structured_new.csv", index=False, encoding='utf-8')


"""
Hadoop
"""
dataset = "Hadoop"
Hadoop_df = pd.read_csv(f"full_dataset/{dataset}/{dataset}_full.log_structured.csv")
Hadoop_items_to_be_replaced = [
    "Task <*> done.",
    "Assigning <*> with <*> to <*>",
    "fetcher#<*> about to shuffle output of map <*>: <*> len: <*> to DISK",
    "MRAppMaster launching normal, non-uberized, multi-container job <*>",
    "Extract jar:<*> to <*>",
    "Saved output of task <*> to <*>",
    "Communication exception: java.io.IOException: Failed on local exception: java.io.IOException: An existing connection was forcibly closed by the remote host; Host Details : local host is: <*> destination host is: <*>",
    "Failed to renew lease for <*> for <*> seconds. Will retry shortly ...",
    "Diagnostics report from <*>: java.io.IOException: Spill failed"
    ]
Hadoop_new_items = [
    "Task '<*>' done.",
    "Assigning <*> with <*> to fetcher#<*>",
    "fetcher#<*> about to shuffle output of map <*> decomp: <*> len: <*> to DISK",
    "MRAppMaster launching normal, non-uberized, multi-container job <*>.",
    "Extract jar:file:<*> to <*>",
    "Saved output of task '<*>' to <*>",
    'Communication exception: java.io.IOException: Failed on local exception: java.io.IOException: An existing connection was forcibly closed by the remote host; Host Details : local host is: "<*>"; destination host is: "<*>":<*>;',
    "Failed to renew lease for [<*>] for <*> seconds. Will retry shortly ...",
    "Diagnostics report from <*>: Error: java.io.IOException: Spill failed"
    ]
for old, new in zip(Hadoop_items_to_be_replaced,Hadoop_new_items):
    Hadoop_df.loc[Hadoop_df["EventTemplate"] == old, "EventTemplate"] = new
Hadoop_df.to_csv(f"full_dataset/{dataset}/{dataset}_full.log_structured_new.csv", index=False, encoding='utf-8')


"""
HDFS
"""
dataset = "HDFS"
HDFS_df = pd.read_csv(f"full_dataset/{dataset}/{dataset}_full.log_structured.csv")
HDFS_items_to_be_replaced = [
    "BLOCK* NameSystem.allocateBlock: <*> <*>",
    "writeBlock <*> received exception java.net.SocketTimeoutException: <*> millis timeout while waiting for channel to be ready for <*> ch : java.nio.channels.SocketChannel[connected local=<*> remote=<*>]"
    ]
HDFS_new_items = [
    'BLOCK* NameSystem.allocateBlock: <*>. <*>',
    "writeBlock <*> received exception java.net.SocketTimeoutException: <*> millis timeout while waiting for channel to be ready for read. ch : java.nio.channels.SocketChannel[connected local=<*> remote=<*>]"
    ]
for old, new in zip(HDFS_items_to_be_replaced,HDFS_new_items):
    HDFS_df.loc[HDFS_df["EventTemplate"] == old, "EventTemplate"] = new
HDFS_df.to_csv(f"full_dataset/{dataset}/{dataset}_full.log_structured_new.csv", index=False, encoding='utf-8')


"""
HealthApp
"""
dataset = "HealthApp"
HealthApp_df = pd.read_csv(f"full_dataset/{dataset}/{dataset}_full.log_structured.csv")
HealthApp_items_to_be_replaced = [
    "REPORT : <*>",
    "new date =<*>, type=<*>,old=<*>",
    "aggregateHiHealthData() time = <*> totalTime = <*>",
    "readHiHealthData() end time = <*> totalTime = <*>",
    "readHiHealthData() readOption = HiDataReadOption{startTime=<*>, endTime=<*>, type=<*>, count = <*>, sortOrder=<*>, readType=<*>, alignType=<*>}"
    ]
HealthApp_new_items = [
    "REPORT : <*> <*> <*> <*>",
    "new date =<*>, type=<*>,<*>,old=<*>",
    "aggregateHiHealthData() time = <*>, totalTime = <*>",
    "readHiHealthData() end time = <*>, totalTime = <*>",
    "readHiHealthData() readOption = HiDataReadOption{startTime=<*>, endTime=<*>, type=[<*>, <*>], count = <*>, sortOrder=<*>, readType=<*>, alignType=<*>}"
    ]
for old, new in zip(HealthApp_items_to_be_replaced,HealthApp_new_items):
    HealthApp_df.loc[HealthApp_df["EventTemplate"] == old, "EventTemplate"] = new
HealthApp_df.to_csv(f"full_dataset/{dataset}/{dataset}_full.log_structured_new.csv", index=False, encoding='utf-8')


"""
HPC
"""
dataset = "HPC"
HPC_df = pd.read_csv(f"full_dataset/{dataset}/{dataset}_full.log_structured.csv")
HPC_items_to_be_replaced = [
    "PSU status (<*> <*>)",
    "Temperature <*> exceeds warning threshold"
    ]
HPC_new_items = [
    "PSU status ( <*> <*> )",
    "Temperature (<*>) exceeds warning threshold"
    ]
for old, new in zip(HPC_items_to_be_replaced, HPC_new_items):
    HPC_df.loc[HPC_df["EventTemplate"] == old, "EventTemplate"] = new
HPC_df.to_csv(f"full_dataset/{dataset}/{dataset}_full.log_structured_new.csv", index=False, encoding='utf-8')

"""
Linux
"""
dataset = "Linux"
Linux_df = pd.read_csv(f"full_dataset/{dataset}/{dataset}_full.log_structured.csv")
Linux_items_to_be_replaced = [
    "couldn't add command channel <*> not found",
    "creating device node <*>"
    ]
Linux_new_items = [
    "couldn't add command channel <*>: not found",
    "creating device node '<*>'"
    ]
for old, new in zip(Linux_items_to_be_replaced, Linux_new_items):
    Linux_df.loc[Linux_df["EventTemplate"] == old, "EventTemplate"] = new
Linux_df.to_csv(f"full_dataset/{dataset}/{dataset}_full.log_structured_new.csv", index=False, encoding='utf-8')


"""
Mac
"""
dataset = "Mac"
Mac_df = pd.read_csv(f"full_dataset/{dataset}/{dataset}_full.log_structured.csv")
Mac_items_to_be_replaced = [
    '<*>: <*> country code set to <*>',
    '-> DNSServiceGetAddrInfo <*> v4v6 <*>. PID[<*>]()',
    ]
Mac_new_items = [
    "<*>: <*> country code set to '<*>'.",
    '-> DNSServiceGetAddrInfo <*> <*> v4v6 <*> PID[<*>]()',
    ]
for old, new in zip(Mac_items_to_be_replaced, Mac_new_items):
    Mac_df.loc[Mac_df["EventTemplate"] == old, "EventTemplate"] = new
Mac_df.to_csv(f"full_dataset/{dataset}/{dataset}_full.log_structured_new.csv", index=False, encoding='utf-8')



"""
OpenSSH
"""

"""
OpenStack
"""
dataset = "OpenStack"
OpenStack_df = pd.read_csv(f"full_dataset/{dataset}/{dataset}_full.log_structured.csv")
OpenStack_items_to_be_replaced = [
    '<*> "GET <*> <*>" status: <*> len: <*> time: <*>',
    '<*> "POST <*> <*>" status: <*> len: <*> time: <*>',
    '<*> "DELETE <*> <*>" status: <*> len: <*> time: <*>'
    ]
OpenStack_new_items = [
    '<*> "GET <*> HTTP/1.1" status: <*> len: <*> time: <*>',
    '<*> "POST <*> HTTP/1.1" status: <*> len: <*> time: <*>',
    '<*> "DELETE <*> HTTP/1.1" status: <*> len: <*> time: <*>'
    ]
for old, new in zip(OpenStack_items_to_be_replaced, OpenStack_new_items):
    OpenStack_df.loc[OpenStack_df["EventTemplate"] == old, "EventTemplate"] = new
OpenStack_df.to_csv(f"full_dataset/{dataset}/{dataset}_full.log_structured_new.csv", index=False, encoding='utf-8')

# """
# Proxifier
# """
# dataset = "Proxifier"
# Proxifier_df = pd.read_csv(f"full_dataset/{dataset}/{dataset}_full.log_structured.csv")
# Proxifier_items_to_be_replaced = [
   
#     ]
# Proxifier_new_items = [
    
#     ]
# for old, new in zip(Proxifier_items_to_be_replaced, Proxifier_new_items):
#     Proxifier_df.loc[HealthApp_df["EventTemplate"] == old, "EventTemplate"] = new
# Proxifier_df.to_csv(f"full_dataset/{dataset}/{dataset}_full.log_structured_new.csv", index=False, encoding='utf-8')


# """
# Spark
# """

"""
Thunderbird
"""
dataset = "Thunderbird"
Thunderbird_df = pd.read_csv(f"full_dataset/{dataset}/{dataset}_full.log_structured.csv")
Thunderbird_items_to_be_replaced = [
    "My unqualified host name <*> unknown; sleeping for retry",
    "synchronized to <*> stratum <*>",
    "DHCPREQUEST for <*> from <*> via <*>: unknown lease <*>.",
    "DHCPREQUEST for <*> from <*> via <*>",
    "ACPI: PCI interrupt <*> -> GSI <*> (level, <*>) -> IRQ <*>",
    "IOAPIC[<*>]: apic_id <*>, version <*>, address <*>, GSI <*>",
    "(<*>) CMD (<*>)"
    ]
Thunderbird_new_items = [
    "My unqualified host name (<*>) unknown; sleeping for retry",
    "synchronized to <*>, stratum <*>",
    "DHCPREQUEST for <*> (<*>) from <*> via <*>: unknown lease <*>.",
    "DHCPREQUEST for <*> (<*>) from <*> via <*>",
    "ACPI: PCI interrupt <*>[<*>] -> GSI <*> (level, <*>) -> IRQ <*>",
    "IOAPIC[<*>]: apic_id <*>, version <*>, address <*>, GSI <*>-<*>",
    "(<*>) CMD (<*> <*>)"
    ]
for old, new in zip(Thunderbird_items_to_be_replaced, Thunderbird_new_items):
    Thunderbird_df.loc[Thunderbird_df["EventTemplate"] == old, "EventTemplate"] = new
Thunderbird_df.to_csv(f"full_dataset/{dataset}/{dataset}_full.log_structured_new.csv", index=False, encoding='utf-8')

"""
Zookeeper
"""
dataset = "Zookeeper"
Zookeeper_df = pd.read_csv(f"full_dataset/{dataset}/{dataset}_full.log_structured.csv")
Zookeeper_items_to_be_replaced = [
    "Server environment:zookeeper.version=<*>, built on <*> GMT",
    "Expiring session <*> timeout of <*> exceeded",
    ]
Zookeeper_new_items = [
    "Server environment:zookeeper.version=<*>, built on <*>",
    "Expiring session <*>, timeout of <*> exceeded"
    ]
for old, new in zip(Zookeeper_items_to_be_replaced, Zookeeper_new_items):
    Zookeeper_df.loc[Zookeeper_df["EventTemplate"] == old, "EventTemplate"] = new
Zookeeper_df.to_csv(f"full_dataset/{dataset}/{dataset}_full.log_structured_new.csv", index=False, encoding='utf-8')