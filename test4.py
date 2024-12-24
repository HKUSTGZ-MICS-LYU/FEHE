import src.filedata as fd
import numpy as np
import tenseal as ts


secret_context = ts.context_from(fd.read_data('src/encrypted/secret_key.txt')[0])


new_results_proto = fd.read_data('src/encrypted/data_encrypted_0.txt')
new_results = []
for new_result_proto in new_results_proto:
    new_result = ts.lazy_ckks_tensor_from(new_result_proto)
    new_result.link_context(secret_context)
    new_result = np.array(new_result.decrypt())
    new_result = new_result.tolist()
    new_results.append(new_result.raw)

print(new_results)