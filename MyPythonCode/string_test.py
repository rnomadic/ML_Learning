dydb_table_name = "RoyTest"
service_name = "mmtdata-devops"
DeliveryStreamName = service_name + "-" + dydb_table_name.lower()
print(DeliveryStreamName)

line = 'Kong Panda'
index = line.find('Panda')
output_line = line[:index] + 'Fu ' + line[index:]
print(output_line)

DeliveryStreamName='mmtdata-devops-{}-stream'.format(dydb_table_name.lower())
print(DeliveryStreamName)