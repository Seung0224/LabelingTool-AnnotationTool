// TrainingServices_AllInOne.cs
// Consolidated single-file services for training/export pipeline.
// Place this file anywhere in your project (e.g., /Services).
// Requires reference: System.Management

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Management;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace SmartLabelingApp
{
    internal static class TrainingConfig
    {
        public const int Epochs = 20;   // GPU용 epoch
        
        public const int ImgSize = 512;
        public const int BatchGpu = 8;
        public const int BatchCpu = 4;
        public const string DeviceGpu = "0";
        public const string DeviceCpu = "cpu";
    }

    // ============================
    // Process helpers (Exec & Logs)
    // ============================
    internal static class ProcessRunner
    {
        internal static int RunProcess(string fileName, string arguments, string workingDirectory)
        {
            var psi = CreatePsi(fileName, arguments, workingDirectory);
            using (var p = Process.Start(psi))
            {
                p.OutputDataReceived += (s, e) => { if (e.Data != null) Trace.WriteLine(e.Data); };
                p.ErrorDataReceived += (s, e) => { if (e.Data != null) Trace.WriteLine(e.Data); };
                p.BeginOutputReadLine();
                p.BeginErrorReadLine();
                p.WaitForExit();
                return p.ExitCode;
            }
        }

        internal static int RunProcessProgressPercentAware(
    string fileName,
    string arguments,
    string workingDirectory,
    int startPct,
    int endPct,
    Action<int, string> progress,
    string phaseLabel,
    IDictionary<string, string> extraEnv = null)
        {
            if (endPct < startPct) endPct = startPct;
            int current = Math.Max(0, Math.Min(99, startPct));
            int upper = Math.Max(current, Math.Min(100, endPct));

            var psi = CreatePsi(fileName, arguments, workingDirectory);
            if (extraEnv != null)
                foreach (var kv in extraEnv) psi.EnvironmentVariables[kv.Key] = kv.Value;

            var rxPercent = new System.Text.RegularExpressions.Regex(@"\b(\d{1,3})\s?%\b",
                System.Text.RegularExpressions.RegexOptions.Compiled);
            var sw = Stopwatch.StartNew();
            long lastOutputMs = 0;
            int lastReported = current;

            using (var p = Process.Start(psi))
            {
                p.OutputDataReceived += (s, e) =>
                {
                    if (e.Data == null) return;
                    Trace.WriteLine(e.Data);
                    lastOutputMs = sw.ElapsedMilliseconds;

                    var m = rxPercent.Match(e.Data);
                    if (m.Success && int.TryParse(m.Groups[1].Value, out int pct))
                    {
                        pct = Math.Max(0, Math.Min(100, pct));
                        int mapped = startPct + (int)Math.Round((pct / 100.0) * (endPct - startPct));
                        if (mapped > lastReported)
                        {
                            lastReported = mapped;
                            progress?.Invoke(mapped, phaseLabel + $"... {pct}%");
                        }
                    }
                };
                p.ErrorDataReceived += (s, e) => { if (e.Data != null) Trace.WriteLine(e.Data); };

                p.BeginOutputReadLine();
                p.BeginErrorReadLine();

                // 하트비트: 무출력 1.5초 이상이면 1%씩 상향(upper-1 한도)
                while (!p.WaitForExit(100))
                {
                    if (sw.ElapsedMilliseconds - lastOutputMs > 1500 && lastReported < upper - 1)
                    {
                        lastReported++;
                        progress?.Invoke(lastReported, phaseLabel + "...");
                        lastOutputMs = sw.ElapsedMilliseconds;
                    }
                }

                progress?.Invoke(upper, phaseLabel + " 완료");
                return p.ExitCode;
            }
        }


        internal static int RunProcess(string fileName, string arguments, string workingDirectory,
                                       IDictionary<string, string> extraEnv)
        {
            var psi = CreatePsi(fileName, arguments, workingDirectory);
            if (extraEnv != null)
                foreach (var kv in extraEnv) psi.EnvironmentVariables[kv.Key] = kv.Value;

            using (var p = Process.Start(psi))
            {
                p.OutputDataReceived += (s, e) => { if (e.Data != null) Trace.WriteLine(e.Data); };
                p.ErrorDataReceived += (s, e) => { if (e.Data != null) Trace.WriteLine(e.Data); };
                p.BeginOutputReadLine();
                p.BeginErrorReadLine();
                p.WaitForExit();
                return p.ExitCode;
            }
        }

        internal static int RunProcessProgress(
            string fileName,
            string arguments,
            string workingDirectory,
            int startPct,
            int endPct,
            Action<int, string> progress,
            string phaseLabel)
        {
            if (endPct < startPct) endPct = startPct;
            int current = Math.Max(0, Math.Min(99, startPct));
            int upper = Math.Max(current, Math.Min(100, endPct));

            var psi = CreatePsi(fileName, arguments, workingDirectory);

            using (var p = Process.Start(psi))
            {
                p.OutputDataReceived += (s, e) => { if (e.Data != null) Trace.WriteLine(e.Data); };
                p.ErrorDataReceived += (s, e) => { if (e.Data != null) Trace.WriteLine(e.Data); };
                p.BeginOutputReadLine();
                p.BeginErrorReadLine();

                var sw = Stopwatch.StartNew();
                int lastBumpMs = 0;
                int bumpIntervalMs = 250;
                while (!p.WaitForExit(100))
                {
                    if (current < upper - 1 && sw.ElapsedMilliseconds - lastBumpMs >= bumpIntervalMs)
                    {
                        current += 1;
                        lastBumpMs = (int)sw.ElapsedMilliseconds;
                        progress?.Invoke(current, phaseLabel + "...");
                    }
                }

                progress?.Invoke(upper, phaseLabel + " 완료");
                return p.ExitCode;
            }
        }

        internal static int RunProcessProgress(
            string fileName,
            string arguments,
            string workingDirectory,
            int startPct,
            int endPct,
            Action<int, string> progress,
            string phaseLabel,
            IDictionary<string, string> extraEnv)
        {
            if (endPct < startPct) endPct = startPct;
            int current = Math.Max(0, Math.Min(99, startPct));
            int upper = Math.Max(current, Math.Min(100, endPct));

            var psi = CreatePsi(fileName, arguments, workingDirectory);
            if (extraEnv != null)
                foreach (var kv in extraEnv) psi.EnvironmentVariables[kv.Key] = kv.Value;

            using (var p = Process.Start(psi))
            {
                p.OutputDataReceived += (s, e) => { if (e.Data != null) Trace.WriteLine(e.Data); };
                p.ErrorDataReceived += (s, e) => { if (e.Data != null) Trace.WriteLine(e.Data); };
                p.BeginOutputReadLine();
                p.BeginErrorReadLine();

                var sw = Stopwatch.StartNew();
                int lastBumpMs = 0, bumpIntervalMs = 250;
                while (!p.WaitForExit(100))
                {
                    if (current < upper - 1 && sw.ElapsedMilliseconds - lastBumpMs >= bumpIntervalMs)
                    {
                        current += 1;
                        lastBumpMs = (int)sw.ElapsedMilliseconds;
                        progress?.Invoke(current, phaseLabel + "...");
                    }
                }

                progress?.Invoke(upper, phaseLabel + " 완료");
                return p.ExitCode;
            }
        }

        internal static int RunProcessCaptureToFile(string fileName, string arguments, string workingDirectory, string logPath)
        {
            var psi = CreatePsi(fileName, arguments, workingDirectory);
            var sb = new StringBuilder();
            using (var p = Process.Start(psi))
            {
                p.OutputDataReceived += (s, e) => { if (e.Data != null) sb.AppendLine(e.Data); };
                p.ErrorDataReceived += (s, e) => { if (e.Data != null) sb.AppendLine(e.Data); };
                p.BeginOutputReadLine();
                p.BeginErrorReadLine();
                p.WaitForExit();
                try { File.WriteAllText(logPath, sb.ToString()); } catch { }
                return p.ExitCode;
            }
        }

        private static ProcessStartInfo CreatePsi(string fileName, string arguments, string workingDirectory)
        {
            return new ProcessStartInfo
            {
                FileName = fileName,
                Arguments = arguments,
                WorkingDirectory = string.IsNullOrEmpty(workingDirectory) ? Environment.CurrentDirectory : workingDirectory,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };
        }
    }

    // ======================
    // GPU detection helpers
    // ======================
    internal static class GpuDetector
    {
        internal static bool CanUseCudaForKernels(string pythonExe, string workingDir)
        {
            // exit codes: 0 = cuda usable, 2 = torch installed but no cuda, 1 = torch not installed, 3 = unexpected error
            // 주의: -c 원라이너에서 try:는 문법적으로 불가 → 예외 없이 분기 처리
            string cmd =
                "-c \"import sys; import importlib, importlib.util; " +
                "m=importlib.util.find_spec('torch'); " +
                "t=importlib.import_module('torch') if m else None; " +
                "sys.exit(0 if (t and hasattr(t,'cuda') and t.cuda.is_available()) else (2 if m else 1))\"";

            int code = ProcessRunner.RunProcess(pythonExe, cmd, workingDir);
            return code == 0;
        }
        internal static bool HasNvidiaGpu()
        {
            try
            {
                using (var searcher = new ManagementObjectSearcher("SELECT Name, AdapterCompatibility FROM Win32_VideoController"))
                using (var results = searcher.Get())
                {
                    foreach (ManagementObject mo in results)
                    {
                        string name = (mo["Name"] as string ?? string.Empty).ToUpperInvariant();
                        string vendor = (mo["AdapterCompatibility"] as string ?? string.Empty).ToUpperInvariant();
                        if (name.Contains("NVIDIA") || vendor.Contains("NVIDIA"))
                            return true;
                    }
                }
            }
            catch { }

            try
            {
                int code = ProcessRunner.RunProcess("nvidia-smi", "-L", Environment.SystemDirectory);
                if (code == 0) return true;
            }
            catch { }

            return false;
        }
    }

    // ==================
    // Disk utils (space)
    // ==================
    internal static class DiskUtils
    {
        internal static void EnsureDiskSpaceOrThrow(string baseDir, IDictionary<string, string> pipEnv, bool wantCuda)
        {
            string tmp = pipEnv != null && pipEnv.ContainsKey("TMP") ? pipEnv["TMP"] : Path.GetTempPath();
            long needBaseGB = wantCuda ? 10 : 5;
            long needTmpGB = wantCuda ? 6 : 3;

            long freeBase = GetFreeBytes(baseDir);
            long freeTmp = GetFreeBytes(tmp);

            if (freeBase < GB(needBaseGB) || freeTmp < GB(needTmpGB))
            {
                string msg =
                    "디스크 여유 공간이 부족합니다.\n" +
                    $"- baseDir({Path.GetPathRoot(Path.GetFullPath(baseDir))}) 여유: {ToGB(freeBase)} GB (필요 ≥ {needBaseGB} GB)\n" +
                    $"- TMP({Path.GetPathRoot(Path.GetFullPath(tmp))}) 여유: {ToGB(freeTmp)} GB (필요 ≥ {needTmpGB} GB)\n\n" +
                    "불필요한 파일 정리 후 다시 시도해 주세요.";
                throw new Exception(msg);
            }
        }

        private static long GetFreeBytes(string anyPathUnderDrive)
        {
            var root = Path.GetPathRoot(Path.GetFullPath(anyPathUnderDrive));
            var di = new DriveInfo(root);
            return di.AvailableFreeSpace;
        }
        private static long GB(long n) => n * 1024L * 1024L * 1024L;
        private static string ToGB(long bytes) => (bytes / (1024.0 * 1024 * 1024)).ToString("0.0");
    }

    // ======================
    // Torch/Ultralytics info
    // ======================
    internal static class TorchInspector
    {
        internal static (bool installed, bool isCuda) CheckTorchVariant(string pythonExe, string workingDir)
        {
            // 0=installed+cuda, 2=installed+cpu, 1=not installed
            string cmd = "-c \"import sys, importlib, importlib.util;" +
                         "m=importlib.util.find_spec('torch');" +
                         "t=None; exec('import torch as t') if m else None;" +
                         "sys.exit(0 if (m and hasattr(t, 'cuda') and t.cuda.is_available()) else (2 if m else 1))\"";

            int code = ProcessRunner.RunProcess(pythonExe, cmd, workingDir);
            if (code == 0) return (true, true);
            if (code == 2) return (true, false);
            return (false, false);
        }

        // TorchInspector.IsUltralyticsInstalled - FIX
        internal static bool IsUltralyticsInstalled(string pythonExe, string workingDir)
        {
            string cmd = "-c \"import sys, importlib, importlib.util;" +
                         "sys.exit(0 if importlib.util.find_spec('ultralytics') else 1)\"";
            int code = ProcessRunner.RunProcess(pythonExe, cmd, workingDir);
            return code == 0;
        }
    }

    // ==============
    // Env setup (venv)
    // ==============
    internal static class EnvSetup
    {
        internal static Task EnsureVenvAndUltralyticsAsync(
            string baseDir,
            string venvDir,
            string pythonExe,
            Action<int, string> progress)
        {
            return Task.Run(() =>
            {
                progress?.Invoke(2, "venv 점검 중...");

                bool needCreate = !(Directory.Exists(venvDir) && File.Exists(pythonExe));
                if (needCreate)
                {
                    progress?.Invoke(5, "venv 생성...");
                    int code = ProcessRunner.RunProcessProgress("py", "-3 -m venv .venv",
                        baseDir, 5, 35, progress, "venv 생성(py -3)", null);
                    if (code != 0)
                    {
                        Trace.WriteLine("[VENV] 'py -3' 실패, 'python'으로 재시도");
                        code = ProcessRunner.RunProcessProgress("python", "-m venv .venv",
                            baseDir, 5, 35, progress, "venv 생성(python)", null);
                        if (code != 0) throw new Exception("venv 생성 실패. Python 3가 PATH에 있는지 확인하세요.");
                    }
                }
                else
                {
                    progress?.Invoke(5, "기존 venv 확인");
                    progress?.Invoke(10, "환경 확인");
                }

                var pipEnv = GetPipEnv(baseDir);

                // pip 업그레이드 (기존 유지)
                int ec = ProcessRunner.RunProcessProgress(
                    pythonExe,
                    "-m pip install --upgrade pip --timeout 120 --retries 2",
                    baseDir, 10, 18, progress, "pip 업그레이드", pipEnv);
                if (ec != 0) throw new Exception("pip 업그레이드 실패");

                progress?.Invoke(20, "GPU 탐지 중...");
                bool wantCuda = GpuDetector.HasNvidiaGpu();
                progress?.Invoke(22, wantCuda ? "NVIDIA GPU 감지 → CUDA용 PyTorch 필요" : "GPU 미탐지 → CPU용 PyTorch 필요");

                progress?.Invoke(24, "PyTorch 상태 점검...");
                var torchState = TorchInspector.CheckTorchVariant(pythonExe, baseDir);
                bool needTorchInstall = !torchState.installed || (wantCuda != torchState.isCuda);

                if (!needTorchInstall)
                {
                    progress?.Invoke(30, $"PyTorch OK({(torchState.isCuda ? "CUDA" : "CPU")}) → 재설치 생략");
                }
                else
                {
                    progress?.Invoke(28, "디스크 용량 확인...");
                    DiskUtils.EnsureDiskSpaceOrThrow(baseDir, pipEnv, wantCuda);

                    // 제거 & 캐시정리는 빠르게
                    progress?.Invoke(30, "기존 PyTorch 제거...");
                    ProcessRunner.RunProcessProgress(pythonExe, "-m pip uninstall -y torch torchvision torchaudio",
                        baseDir, 30, 32, progress, "PyTorch 제거", pipEnv);

                    progress?.Invoke(32, "pip 캐시 정리...");
                    ProcessRunner.RunProcessProgress(pythonExe, "-m pip cache purge",
                        baseDir, 32, 34, progress, "pip 캐시 정리", pipEnv);

                    // 1) 공통 경량 의존성 먼저 (분할 설치)
                    string commonPkgs = "mpmath typing-extensions sympy pillow numpy networkx MarkupSafe fsspec filelock jinja2";
                    ec = ProcessRunner.RunProcessProgressPercentAware(
                        pythonExe,
                        $"-m pip install --upgrade --no-cache-dir --prefer-binary {commonPkgs} --timeout 300 --retries 2 -v",
                        baseDir, 34, 40, progress, "기본 의존성 설치", pipEnv);
                    if (ec != 0) throw new Exception("기본 의존성 설치 실패");

                    // 2) torch (대형 wheel → % 파싱)
                    string torchIdx = wantCuda ? "--index-url https://download.pytorch.org/whl/cu121" : "";
                    ec = ProcessRunner.RunProcessProgressPercentAware(
                        pythonExe,
                        $"-m pip install --upgrade --force-reinstall --no-cache-dir --prefer-binary {torchIdx} torch --timeout 900 --retries 2 -v",
                        baseDir, 40, 58, progress, "PyTorch 설치(torch)", pipEnv);
                    if (ec != 0) throw new Exception("PyTorch(torch) 설치 실패");

                    // 3) torchvision (대형 wheel → % 파싱)
                    ec = ProcessRunner.RunProcessProgressPercentAware(
                        pythonExe,
                        $"-m pip install --upgrade --force-reinstall --no-cache-dir --prefer-binary {torchIdx} torchvision --timeout 900 --retries 2 -v",
                        baseDir, 58, 62, progress, "PyTorch 설치(torchvision)", pipEnv);
                    if (ec != 0) throw new Exception("PyTorch(torchvision) 설치 실패");
                }

                // Ultralytics
                progress?.Invoke(64, "Ultralytics 상태 점검...");
                bool hasUltralytics = TorchInspector.IsUltralyticsInstalled(pythonExe, baseDir);

                if (hasUltralytics)
                {
                    progress?.Invoke(70, "Ultralytics OK → 재설치 생략");
                }
                else
                {
                    progress?.Invoke(70, "Ultralytics 설치...");
                    string ultraCmd1 =
                        "-m pip install --upgrade --no-cache-dir --prefer-binary ultralytics " +
                        "--extra-index-url https://download.pytorch.org/whl/cu121 " +
                        "--timeout 600 --retries 2 -v";

                    ec = ProcessRunner.RunProcessProgressPercentAware(
                        pythonExe, ultraCmd1, baseDir, 70, 88, progress, "ultralytics 설치", pipEnv);

                    if (ec != 0)
                    {
                        progress?.Invoke(88, "pip 캐시 정리...");
                        ProcessRunner.RunProcessProgress(
                            pythonExe, "-m pip cache purge", baseDir, 88, 90, progress, "pip 캐시 정리", pipEnv);

                        progress?.Invoke(90, "ultralytics 재시도(상세 로그)...");
                        string ultraCmd2 =
                            "-m pip install -vvv --no-cache-dir --prefer-binary ultralytics==8.3.190 " +
                            "--extra-index-url https://download.pytorch.org/whl/cu121 " +
                            "--timeout 900 --retries 1";
                        ec = ProcessRunner.RunProcessProgressPercentAware(
                            pythonExe, ultraCmd2, baseDir, 90, 95, progress, "ultralytics 재시도", pipEnv);

                        if (ec != 0)
                        {
                            Trace.WriteLine("[INSTALL] Ultralytics 설치가 최종 실패했습니다. 위의 pip -vvv 로그를 확인하세요.");
                            throw new Exception("ultralytics 설치 실패(Trace 로그 확인)");
                        }
                    }
                }

                progress?.Invoke(100, "환경 준비 완료");
                Trace.WriteLine("[VENV] Ready with PyTorch " + ((needTorchInstall ? wantCuda : torchState.isCuda) ? "CUDA" : "CPU"));
            });
        }

        internal static IDictionary<string, string> GetPipEnv(string baseDir)
        {
            string cacheDir = Path.Combine(baseDir, ".pipcache");
            string tmpDir = Path.Combine(baseDir, ".tmp");
            Directory.CreateDirectory(cacheDir);
            Directory.CreateDirectory(tmpDir);

            var env = new Dictionary<string, string>();
            env["PIP_CACHE_DIR"] = cacheDir;
            env["TMP"] = tmpDir;
            env["TEMP"] = tmpDir;
            return env;
        }
    }


    // =====================
    // YOLO CLI small utils
    // =====================
    internal static class YoloCli
    {
        internal static (string fileName, string argumentsPrefix) GetYoloCli(string yoloExePath, string pythonExePath)
        {
            if (File.Exists(yoloExePath)) return (yoloExePath, "");
            return (pythonExePath, "-m ultralytics");
        }

        internal static string Quote(string path)
        {
            if (string.IsNullOrEmpty(path)) return "\"\"";
            if (path.StartsWith("\"") && path.EndsWith("\"")) return path;
            return "\"" + path + "\"";
        }
    }

    // ======================
    // YOLO training progress
    // ======================
    internal static class YoloTrainer
    {
        internal static Task<int> RunYoloTrainWithEpochProgressAsync(
            string fileName,
            string arguments,
            string workingDirectory,
            Action<int, string> progress,
            int startPct,
            int endPct,
            int? expectedTotalEpochs = null)
        {
            return Task.Run(() =>
            {
                if (endPct < startPct) endPct = startPct;
                int lastReportedPct = Math.Max(0, startPct);
                int lastEpoch = 0;

                // Epoch 라인만 인정: "Epoch 3/20", "epoch: 3/20", "Epoch (3/20)" 등
                var rxEpoch1 = new Regex(@"\b[Ee]poch\s+(\d+)\s*/\s*(\d+)\b", RegexOptions.Compiled);
                var rxEpoch2 = new Regex(@"\b[Ee]poch\s*:\s*(\d+)\s*/\s*(\d+)\b", RegexOptions.Compiled);
                var rxEpoch3 = new Regex(@"\b[Ee]poch\s*\(\s*(\d+)\s*/\s*(\d+)\s*\)", RegexOptions.Compiled);

                int totalEpochs = expectedTotalEpochs.GetValueOrDefault(0);

                var psi = new ProcessStartInfo
                {
                    FileName = fileName,
                    Arguments = arguments,
                    WorkingDirectory = string.IsNullOrEmpty(workingDirectory) ? Environment.CurrentDirectory : workingDirectory,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };

                var sw = Stopwatch.StartNew();
                long lastOutputMs = 0;

                using (var p = Process.Start(psi))
                {
                    p.OutputDataReceived += (s, e) =>
                    {
                        if (e.Data == null) return;
                        Trace.WriteLine(e.Data);
                        lastOutputMs = sw.ElapsedMilliseconds;

                        Match m = rxEpoch1.Match(e.Data);
                        if (!m.Success) m = rxEpoch2.Match(e.Data);
                        if (!m.Success) m = rxEpoch3.Match(e.Data);
                        if (!m.Success) return; // 배치 tqdm(예: "12/1000")는 'Epoch'가 없으므로 무시

                        if (!int.TryParse(m.Groups[1].Value, out int cur)) return;
                        if (!int.TryParse(m.Groups[2].Value, out int totFromLog)) return;

                        // 총 에폭 결정: expected가 있으면 그것을 상한으로 사용
                        int tot = totalEpochs > 0 ? totalEpochs : Math.Max(totFromLog, 1);

                        // 로그 값이 비정상(너무 크거나 작음)일 땐 expected로 교정
                        if (totalEpochs > 0)
                        {
                            if (totFromLog > totalEpochs * 2 || totFromLog < Math.Max(1, totalEpochs / 2))
                                tot = totalEpochs;
                            else
                                tot = totalEpochs; // expected 존재 시 그대로 사용
                        }
                        else
                        {
                            totalEpochs = tot;
                        }

                        cur = Math.Max(0, Math.Min(cur, tot));
                        if (cur <= lastEpoch) return; // 되돌림 방지
                        lastEpoch = cur;

                        double frac = Math.Min(1.0, Math.Max(0.0, (double)lastEpoch / tot));
                        int pct = startPct + (int)Math.Round(frac * (endPct - startPct));

                        if (pct > lastReportedPct)
                        {
                            lastReportedPct = pct;
                            string msg = $"학습 중... epoch {lastEpoch}/{tot} ({(int)(frac * 100)}%)";
                            progress?.Invoke(pct, msg);
                        }
                    };

                    p.ErrorDataReceived += (s, e) => { if (e.Data != null) Trace.WriteLine(e.Data); };

                    p.BeginOutputReadLine();
                    p.BeginErrorReadLine();


                    // 하트비트: 무출력 1.5초 이상 → endPct-1 까지만 미세 증가
                    while (!p.WaitForExit(100))
                    {
                        if (sw.ElapsedMilliseconds - lastOutputMs > 1500 && lastReportedPct < endPct - 1)
                        {
                            lastReportedPct++;

                            double frac = (double)lastEpoch / totalEpochs;
                            string hbMsg = $"학습 중...";

                            progress?.Invoke(lastReportedPct, hbMsg);
                            lastOutputMs = sw.ElapsedMilliseconds;
                        }
                    }

                    progress?.Invoke(endPct, "학습 단계 종료");
                    return p.ExitCode;
                }
            });
        }
    }


    // ==================
    // ZIP extract utils
    // ==================
    internal static class ZipDatasetUtils
    {
        internal static Task ExtractZipWithProgressAsync(string zipPath, string destRoot, Action<int, string> progress)
        {
            return Task.Run(() =>
            {
                progress?.Invoke(2, "기존 Result 폴더 제거");
                if (Directory.Exists(destRoot))
                {
                    try { Directory.Delete(destRoot, true); }
                    catch { System.Threading.Thread.Sleep(200); Directory.Delete(destRoot, true); }
                }
                Directory.CreateDirectory(destRoot);
                progress?.Invoke(5, "압축 해제 준비");

                using (var fs = new FileStream(zipPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                using (var archive = new ZipArchive(fs, ZipArchiveMode.Read))
                {
                    var entries = archive.Entries.Where(e => !string.IsNullOrEmpty(e.FullName)).ToList();
                    var files = entries.Where(e => !string.IsNullOrEmpty(e.Name)).ToList();
                    int total = Math.Max(1, files.Count);
                    int processed = 0;

                    string destRootFull = Path.GetFullPath(destRoot);

                    foreach (var entry in entries)
                    {
                        string targetPath = Path.Combine(destRoot, entry.FullName);
                        string targetFull = Path.GetFullPath(targetPath);
                        if (!targetFull.StartsWith(destRootFull, StringComparison.OrdinalIgnoreCase))
                            throw new IOException("압축 내 경로가 허용 범위를 벗어납니다: " + entry.FullName);

                        if (string.IsNullOrEmpty(entry.Name))
                        {
                            Directory.CreateDirectory(targetFull);
                            continue;
                        }

                        Directory.CreateDirectory(Path.GetDirectoryName(targetFull));
                        using (var src = entry.Open())
                        using (var dst = new FileStream(targetFull, FileMode.Create, FileAccess.Write, FileShare.None))
                        {
                            src.CopyTo(dst);
                        }

                        processed++;
                        int pct = 5 + (int)(80.0 * processed / total);
                        if (pct > 85) pct = 85;
                        progress?.Invoke(pct, $"압축 해제 중... ({processed}/{total})");
                    }
                }

                progress?.Invoke(86, "압축 해제 완료");
            });
        }
    }

    // ==========================
    // data.yaml patching helpers
    // ==========================
    internal static class DataYamlPatcher
    {
        internal static string FindDataYaml(string rootDir)
        {
            var yaml = Directory.EnumerateFiles(rootDir, "data.yaml", SearchOption.AllDirectories).FirstOrDefault();
            if (!string.IsNullOrEmpty(yaml)) return yaml;

            yaml = Directory.EnumerateFiles(rootDir, "*.*", SearchOption.AllDirectories)
                             .FirstOrDefault(p => string.Equals(Path.GetFileName(p), "data.yaml", StringComparison.OrdinalIgnoreCase));
            return yaml ?? string.Empty;
        }

        internal static void ValidateRequiredDirs(string datasetRoot)
        {
            string[] need = { @"images\train", @"images\val", @"labels\train", @"labels\val" };
            foreach (var rel in need)
            {
                var p = Path.Combine(datasetRoot, rel);
                if (!Directory.Exists(p))
                    throw new DirectoryNotFoundException("필수 경로가 없습니다: " + p);
            }
        }

        internal static (int images, int labels) CountPair(string datasetRoot, string imagesRel, string labelsRel)
        {
            var imgDir = Path.Combine(datasetRoot, imagesRel);
            var lblDir = Path.Combine(datasetRoot, labelsRel);
            string[] imgExt = { ".jpg", ".jpeg", ".png", ".bmp", ".webp" };

            int imgCount = Directory.Exists(imgDir)
                ? Directory.EnumerateFiles(imgDir, "*.*", SearchOption.AllDirectories)
                    .Count(f => imgExt.Contains(Path.GetExtension(f).ToLowerInvariant()))
                : 0;

            int lblCount = Directory.Exists(lblDir)
                ? Directory.EnumerateFiles(lblDir, "*.txt", SearchOption.AllDirectories).Count()
                : 0;

            return (imgCount, lblCount);
        }

        internal static void FixDataYamlForExtractedDataset(string yamlPath, string datasetRoot)
        {
            string imgTrain = Path.Combine(datasetRoot, @"images\train");
            string imgVal = Path.Combine(datasetRoot, @"images\val");
            string imgTest = Path.Combine(datasetRoot, @"images\test");

            if (!Directory.Exists(imgTrain) || !Directory.Exists(imgVal))
                throw new Exception("data.yaml 수정 실패: images/train 또는 images/val 폴더가 없습니다.");

            string Norm(string p) => "\"" + p.Replace('\\', '/') + "\"";

            string newPathLine = "path: " + Norm(datasetRoot);
            string newTrainLine = "train: images/train";
            string newValLine = "val: images/val";
            string newTestLine = Directory.Exists(imgTest) ? "test: images/test" : "test: images/val";

            var lines = File.ReadAllLines(yamlPath, Encoding.UTF8).ToList();

            bool hasPath = false, hasTrain = false, hasVal = false, hasTest = false;
            var rxPath = new Regex(@"^\s*path\s*:\s*(.+)$", RegexOptions.IgnoreCase);
            var rxTrain = new Regex(@"^\s*train\s*:\s*(.+)$", RegexOptions.IgnoreCase);
            var rxVal = new Regex(@"^\s*val\s*:\s*(.+)$", RegexOptions.IgnoreCase);
            var rxTest = new Regex(@"^\s*test\s*:\s*(.+)$", RegexOptions.IgnoreCase);

            for (int i = 0; i < lines.Count; i++)
            {
                string s = lines[i];
                if (rxPath.IsMatch(s)) { lines[i] = newPathLine; hasPath = true; continue; }
                if (rxTrain.IsMatch(s)) { lines[i] = newTrainLine; hasTrain = true; continue; }
                if (rxVal.IsMatch(s)) { lines[i] = newValLine; hasVal = true; continue; }
                if (rxTest.IsMatch(s)) { lines[i] = newTestLine; hasTest = true; continue; }
            }

            var insert = new List<string>();
            if (!hasPath) insert.Add(newPathLine);
            if (!hasTrain) insert.Add(newTrainLine);
            if (!hasVal) insert.Add(newValLine);
            if (!hasTest) insert.Add(newTestLine);

            if (insert.Count > 0)
            {
                insert.Reverse();
                foreach (var add in insert) lines.Insert(0, add);
            }

            File.WriteAllLines(yamlPath, lines.ToArray(), Encoding.UTF8);
        }
    }

    // =====================
    // Optional path helper
    // =====================
    internal static class PathHelper
    {
        internal static readonly string PreferredStorageRoot = @"D:\SmartLabelingApp";
        
        internal static string GetWeightsRoot()
        {
            var weightsRoot = Path.Combine(PreferredStorageRoot, "weights");
            Directory.CreateDirectory(weightsRoot);
            return weightsRoot;
        }

        internal static string GetVenvRoot()
        {
            var venvRoot = Path.Combine(PreferredStorageRoot, "venv");
            Directory.CreateDirectory(venvRoot);
            return venvRoot;
        }

        internal static string GetPretrainedStandardPath()
        {
            return Path.Combine(GetWeightsRoot(), "yolo11x-seg.pt");
        }

        internal static string GetPretrainedLegacyPath()
        {
            return Path.Combine(GetWeightsRoot(), "yolo11x-seg.pt");
        }

        internal static string ResolvePretrainedPath()
        {
            string std = GetPretrainedStandardPath();
            if (File.Exists(std)) return std;
            string legacy = GetPretrainedLegacyPath();
            if (File.Exists(legacy)) return legacy;
            return std;
        }
    }
}