import csv

file_name = "file_name.csv"
skip_n = 50

pre, fwd, post, total = [], [], [], []

with open(file_name, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pre.append(float(row["preprocess_ms"]))
        fwd.append(float(row["forward_ms"]))
        post.append(float(row["post_ms"]))
        total.append(float(row["total_ms"]))

pre   = pre[skip_n:]
fwd   = fwd[skip_n:]
post  = post[skip_n:]
total = total[skip_n:]

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

if total:
    print(f"skip   : {skip_n}")
    print(f"count  : {len(total)}")
    print(f"pre    : {mean(pre):.2f} ms")
    print(f"fwd    : {mean(fwd):.2f} ms")
    print(f"post   : {mean(post):.2f} ms")
    print(f"total  : {mean(total):.2f} ms")
else:
    print("no data after skipping")
