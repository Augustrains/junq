$path='D:\junq\dppo\envs\reward_manager.py'
$text=[IO.File]::ReadAllText($path)
if($text -notmatch 'import os'){ $text=$text.Replace("import json`n", "import json`nimport os`n") }
$new=@"
    @staticmethod
    def _merge_rules(base, fragment):
        for key, value in fragment.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                RewardManager._merge_rules(base[key], value)
            else:
                base[key] = deepcopy(value)
        return base

    @classmethod
    def _load_rules(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        fragments = manifest.pop("fragments", None)
        if not fragments:
            return manifest
        merged = dict(manifest)
        base_dir = os.path.dirname(os.path.abspath(path))
        for relative_path in fragments:
            fragment_path = os.path.join(base_dir, str(relative_path))
            with open(fragment_path, "r", encoding="utf-8") as f:
                cls._merge_rules(merged, json.load(f))
        return merged
"@
$pattern='(?s)    @staticmethod\r?\n    def _load_rules\(path\):\r?\n        with open\(path, "r", encoding="utf-8"\) as f:\r?\n            return json\.load\(f\)'
$updated=[regex]::Replace($text,$pattern,$new.TrimEnd(),1)
if($updated -eq $text){throw 'RewardManager loader target not found'}
[IO.File]::WriteAllText($path,$updated,[Text.UTF8Encoding]::new($false))
& 'C:\Users\admin\.conda\envs\afsim-ppo\python.exe' -m py_compile $path