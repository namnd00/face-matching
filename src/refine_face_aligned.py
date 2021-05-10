import os

for mset in ['train', 'test']:
    path_96x112 = f"../datasets/aligned/112x112/{mset}"
    files_96x112 = os.listdir(path_96x112)
    male_96x112 = [f[:-6] for f in files_96x112 if "M" in f]
    female_96x112 = [f[:-7] for f in files_96x112 if "F" in f]

    for p in ["96x112", "150x150", "160x160", "224x224"]:
        path = f"../datasets/aligned/{p}/{mset}"

        files_dir = os.listdir(path)
        print(len(files_96x112), len(files_dir), len(male_96x112), len(female_96x112))
        unknown = []
        for file in files_dir:
            if file in files_96x112:
                continue
            unknown.append(file)

        male_unknown = [(f[:-6], f) for f in unknown if "M" in f]
        female_unknown = [(f[:-7], f) for f in unknown if "F" in f]
        male_refined = []
        female_refined = []
        fail_files = []
        for f1, f2 in male_unknown:
            if f1 in female_96x112:
                old_file = os.path.join(path, f2)
                new_file = os.path.join(path, f1 + ".F.jpg")
                os.rename(old_file, new_file)
            else:
                fail_files.append(f2)
        for f1, f2 in female_unknown:
            if f1 in male_96x112:
                old_file = os.path.join(path, f2)
                new_file = os.path.join(path, f1 + ".M.jpg")
                os.rename(old_file, new_file)
            else:
                fail_files.append(f2)
        for f in fail_files:
            path_f = os.path.join(path, f)
            os.remove(path_f)
