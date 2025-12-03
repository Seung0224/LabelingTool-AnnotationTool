using System;
using System.Linq;
using System.Drawing;
using System.Drawing.Imaging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace SmartLabelingApp

{    /// <summary>
     /// 이미지 전처리 공통 모듈:
     /// - 레터박스(letterbox)로 net×net 24bpp 캔버스에 중앙 정렬
     /// - [1,3,net,net] NCHW float 텐서(R,G,B 순)로 0~1 정규화 채우기
     /// </summary>
     /// 
     ///FillTensorFromBitmap
     // 1) 원본 크기(WxH)에서 네트 입력(net x net)에 맞게 비율 유지(scale)로 줄입니다.
     // 2) 줄인 결과를 검은 바탕의 정사각형 캔버스(net x net) 중앙에 그립니다(padX/padY).
     // 3) 픽셀을 읽어 [R,G,B] 채널 순서로 0~1로 정규화하여 _inBuf에 채웁니다.
     //    (왜? 딥러닝 모델이 [1,3,net,net] float 텐서를 입력으로 기대하기 때문)
     // 4) 이후 박스/마스크를 원본 좌표로 되돌리기 위해 scale/padX/padY/resized를 반환합니다.
    
    public static class Preprocess
    {
        public const int DefaultNet = 512;

        /// <summary>
        /// 원본 Bitmap을 네트 입력(net×net)에 맞게 레터박스 후, outNCHW 버퍼([1,3,net,net])에 채웁니다.
        /// R,G,B 채널 순서로 0~1 정규화된 값을 기록합니다.
        /// </summary>
        /// <param name="src">원본 비트맵</param>
        /// <param name="net">네트 입력 한 변(예: 640)</param>
        /// <param name="outNCHW">출력 버퍼([1,3,net,net]) — 호출자가 할당</param>
        /// <param name="scale">축소 배율</param>
        /// <param name="padX">좌측 패딩(px)</param>
        /// <param name="padY">상단 패딩(px)</param>
        /// <param name="resized">스케일 적용 후(패딩 전) 크기</param>
        public static void FillTensorFromBitmap(Bitmap src, int net, float[] outNCHW, out float scale, out int padX, out int padY, out Size resized)
        {
            if (src == null) throw new ArgumentNullException(nameof(src));
            if (outNCHW == null) throw new ArgumentNullException(nameof(outNCHW));
            int need = 1 * 3 * net * net;
            if (outNCHW.Length != need)
                throw new ArgumentException($"outNCHW length must be {need} for [1,3,{net},{net}].", nameof(outNCHW));

            int W = src.Width, H = src.Height;
            scale = Math.Min((float)net / W, (float)net / H);
            int rw = (int)Math.Round(W * scale);
            int rh = (int)Math.Round(H * scale);
            padX = (net - rw) / 2;
            padY = (net - rh) / 2;
            resized = new Size(rw, rh);

            // 24bpp RGB 캔버스에 레터박싱
            using (var tmp = new Bitmap(net, net, PixelFormat.Format24bppRgb))
            using (var g = Graphics.FromImage(tmp))
            {
                g.Clear(Color.Black);
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.Bilinear;
                g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.Half;
                g.DrawImage(src, padX, padY, rw, rh);

                var rect = new Rectangle(0, 0, net, net);
                var bd = tmp.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
                try
                {
                    unsafe
                    {
                        byte* basePtr = (byte*)bd.Scan0;
                        int stride = bd.Stride;
                        float inv255 = 1f / 255f;
                        int plane = net * net;

                        for (int y = 0; y < net; y++)
                        {
                            byte* row = basePtr + y * stride;
                            for (int x = 0; x < net; x++)
                            {
                                int idx = y * net + x;
                                // BGR(24bpp) → NCHW(R,G,B)
                                byte b = row[x * 3 + 0];
                                byte gch = row[x * 3 + 1];
                                byte r = row[x * 3 + 2];
                                outNCHW[0 * plane + idx] = r * inv255;
                                outNCHW[1 * plane + idx] = gch * inv255;
                                outNCHW[2 * plane + idx] = b * inv255;
                            }
                        }
                    }
                }
                finally { tmp.UnlockBits(bd); }
            }
        }


        /// <summary>
        /// Ensure NCHW float buffer of size [1,3,net,net].
        /// Returns true if a new buffer was allocated.
        /// </summary>
        public static bool EnsureNchwBuffer(int net, ref float[] buf)
        {
            int need = 1 * 3 * net * net;
            if (buf == null || buf.Length != need)
            {
                buf = new float[need];
                return true;
            }
            return false;
        }


        /// <summary>
        /// ONNX-specific input preparation moved from YoloSegOnnx.
        /// Ensures ONNX input buffer/tensor/NamedOnnxValue are ready for the given net.
        /// Returns true if internal objects were recreated.
        /// </summary>
        public static bool EnsureOnnxInput(
        InferenceSession session,
        int net,
        ref string inputName,
        ref int curNet,
        ref float[] inBuf,
        ref DenseTensor<float> tensor,
        ref NamedOnnxValue nov)
        {
            if (tensor != null && curNet == net)
                return false; // reuse existing

            // Input name (first input)
            inputName = session.InputMetadata.Keys.First();
            
            // Backing buffer + tensor view
            inBuf = new float[1 * 3 * net * net];
            tensor = new DenseTensor<float>(inBuf, new[] { 1, 3, net, net });

            // Named input for Run()
            nov = NamedOnnxValue.CreateFromTensor(inputName, tensor);

            curNet = net;
            return true;
        }
    }
}
