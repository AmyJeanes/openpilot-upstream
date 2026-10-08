// Paint analysis of one texture, in texture space: which texels are road paint (white or yellow), and the paint's
// line bands (a line's across position, width, colour and painted stretches along it) for the exact line geometry.
using System;
using System.Collections.Generic;
using System.Linq;

public class Band
{
  public char Axis = 'v';   // the line runs along v (at u = Pos) or along u (at v = Pos)
  public float Pos, Width;  // across, in texture units
  public byte Colour;       // 1 white, 2 yellow
  public bool Wraps;        // across position repeats every 1 (road-surface textures tile; atlas cells don't)
  public List<(float a, float b)> Painted = new(); // painted stretches along the line, within [0,1), repeating
}

public class Tex
{
  public int Id;
  public string Name;
  public bool Decal;
  public int Cls;
  public int W, H;
  public byte[] Paint;  // per texel: 0 none, 1 white, 2 yellow
  public byte[] Alpha;  // decals only
  public bool SurfaceDecal;  // a road surface drawn with a decal shader: it is the road where it is opaque
  public List<Band> Bands = new();
}

public static class Paint
{
  static bool Yellowish(float r, float g, float b) => r - b > 0.2f && b < 0.65f * r && g > 0.55f * r;

  static float Median(float[] a)
  {
    var c = (float[])a.Clone();
    Array.Sort(c);
    return c[c.Length / 2];
  }

  // Road-surface textures bake their lines in as bright bands along v. A column is in a band when the mean of its
  // brightest 30% stands out from the texture's other columns (so worn lines and dashes covering a third count); the
  // painted rows of a band are where it stays bright (dashes), and its colour is the majority of its painted texels.
  public static void Surface(Tex t, byte[] bgra)
  {
    int W = t.W, H = t.H;
    var m = new float[W * H];
    var yel = new bool[W * H];
    for (int i = 0; i < W * H; i++)
    {
      float b = bgra[4 * i] / 255f, g = bgra[4 * i + 1] / 255f, r = bgra[4 * i + 2] / 255f;
      m[i] = Math.Max(r, Math.Max(g, b));
      yel[i] = Yellowish(r, g, b);
    }
    float med = Median(m);
    var top = new float[W];
    var col = new float[H];
    int from = (int)(H * 0.7);
    for (int j = 0; j < W; j++)
    {
      for (int r = 0; r < H; r++) col[r] = m[r * W + j] - med;
      Array.Sort(col);
      float s = 0;
      for (int r = from; r < H; r++) s += col[r];
      top[j] = s / (H - from);
    }
    float baseline = Median(top);
    var band = top.Select(v => v > baseline + 0.07f).ToArray();
    t.Paint = new byte[W * H];
    int start = Array.IndexOf(band, false);
    if (start < 0) return;
    for (int k = 1; k <= W; k++)
    {
      int j0 = (start + k) % W;
      if (!band[j0] || band[(j0 - 1 + W) % W]) continue;
      int len = 0;
      while (len < W && band[(j0 + len) % W]) len++;
      if (len < 2 || len >= W * 0.25) continue;
      var rows = new float[H];
      float peak = float.MinValue;
      for (int c = 0; c < len; c++) peak = Math.Max(peak, top[(j0 + c) % W]);
      for (int r = 0; r < H; r++)
      {
        float s = 0;
        for (int c = 0; c < len; c++) s += m[r * W + (j0 + c) % W] - med;
        rows[r] = s / len;
      }
      int kern = Math.Max(H / 24, 3);
      var sm = new float[H];
      for (int r = 0; r < H; r++)
      {
        float s = 0;
        for (int q = -kern / 2; q < kern - kern / 2; q++) s += rows[((r + q) % H + H) % H];
        sm[r] = s / kern;
      }
      float thr = baseline + 0.3f * (peak - baseline);
      var painted = sm.Select(v => v > thr).ToArray();
      float frac = painted.Count(p => p) / (float)H;
      if (frac > 0.7f) Array.Fill(painted, true);
      else if (frac < 0.08f) continue;
      int ny = 0, n = 0;
      for (int r = 0; r < H; r++)
        if (painted[r])
          for (int c = 0; c < len; c++) { n++; if (yel[r * W + (j0 + c) % W]) ny++; }
      byte colour = (byte)(ny * 2 > n ? 2 : 1);
      for (int r = 0; r < H; r++)
        if (painted[r])
          for (int c = 0; c < len; c++) t.Paint[r * W + (j0 + c) % W] = colour;
      var b = new Band { Axis = 'v', Pos = ((j0 + len / 2f) / W) % 1f, Width = len / (float)W, Colour = colour, Wraps = true };
      b.Painted = Runs(painted);
      t.Bands.Add(b);
    }
  }

  // painted stretches of a periodic row mask, as [a, b) fractions; one wrapping across the seam is split in two
  static List<(float, float)> Runs(bool[] p)
  {
    var res = new List<(float, float)>();
    int H = p.Length;
    if (p.All(x => x)) { res.Add((0f, 1f)); return res; }
    for (int r = 0; r < H; r++)
    {
      if (!p[r] || (r > 0 && p[r - 1])) continue;
      int e = r;
      while (e < H && p[e]) e++;
      res.Add((r / (float)H, e / (float)H));
    }
    return res;
  }

  // A decal drawn as strips of line paint (a sheet of white and yellow lines, each strip mapped onto its own road
  // line) rather than a road surface: its opaque texels are mostly paint coloured and the opacity changes in whole
  // columns (lines along v) or rows (along u). Its lines are in the alpha, not in the colour, which is flat underneath.
  public static (bool v, bool u) Strips(byte[] bgra, int W, int H)
  {
    var op = new bool[W * H];
    int nop = 0, npaint = 0;
    for (int i = 0; i < W * H; i++)
    {
      if (bgra[4 * i + 3] <= 128) continue;
      op[i] = true; nop++;
      float b = bgra[4 * i] / 255f, g = bgra[4 * i + 1] / 255f, r = bgra[4 * i + 2] / 255f;
      float mx = Math.Max(r, Math.Max(g, b)), mn = Math.Min(r, Math.Min(g, b));
      if (Yellowish(r, g, b) || (mx > 0.5f && mx - mn < 0.15f)) npaint++;
    }
    float f = nop / (float)(W * H);
    if (f < 0.03f || f > 0.75f || npaint < 0.7f * nop) return (false, false);
    float Split(bool cols)
    {
      int A = cols ? W : H, B = cols ? H : W, ok = 0;
      for (int a = 0; a < A; a++)
      {
        int n = 0;
        for (int b = 0; b < B; b++) if (op[cols ? b * W + a : a * W + b]) n++;
        float fr = n / (float)B;
        if (fr > 0.6f || fr < 0.15f) ok++;
      }
      return ok / (float)A;
    }
    return (Split(true) > 0.7f, Split(false) > 0.7f);
  }

  // Decals: opaque, bright texels are paint (opened by a 3x3 square so specks drop out).
  public static void Decal(Tex t, byte[] bgra)
  {
    int W = t.W, H = t.H;
    var cand = new bool[W * H];
    var yel = new bool[W * H];
    t.Alpha = new byte[W * H];
    for (int i = 0; i < W * H; i++)
    {
      float b = bgra[4 * i] / 255f, g = bgra[4 * i + 1] / 255f, r = bgra[4 * i + 2] / 255f;
      t.Alpha[i] = bgra[4 * i + 3];
      cand[i] = bgra[4 * i + 3] > 128 && Math.Max(r, Math.Max(g, b)) > 0.4f;
      yel[i] = Yellowish(r, g, b);
    }
    var er = Morph(cand, W, H, false);
    var op = Morph(er, W, H, true);
    t.Paint = new byte[W * H];
    for (int i = 0; i < W * H; i++) if (op[i]) t.Paint[i] = (byte)(yel[i] ? 2 : 1);
  }

  static bool[] Morph(bool[] a, int W, int H, bool dilate)
  {
    var o = new bool[W * H];
    for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++)
      {
        bool v = !dilate;
        for (int dy = -1; dy <= 1; dy++)
          for (int dx = -1; dx <= 1; dx++)
          {
            int xx = Math.Clamp(x + dx, 0, W - 1), yy = Math.Clamp(y + dy, 0, H - 1);
            if (dilate) v |= a[yy * W + xx]; else v &= a[yy * W + xx];
          }
        o[y * W + x] = v;
      }
    return o;
  }

  // Line bands of a decal atlas cell whose strips run along `axis`: columns (or rows) mostly painted within the cell.
  public static void DecalBands(Tex t, float u0, float u1, float v0, float v1, char axis, bool wraps = false)
  {
    int W = t.W, H = t.H;
    int a0, a1, b0, b1, A, B;  // across range, along range (texels), across/along sizes
    if (axis == 'v') { a0 = (int)(u0 * W); a1 = (int)Math.Ceiling(u1 * W); b0 = (int)(v0 * H); b1 = (int)Math.Ceiling(v1 * H); A = W; B = H; }
    else { a0 = (int)(v0 * H); a1 = (int)Math.Ceiling(v1 * H); b0 = (int)(u0 * W); b1 = (int)Math.Ceiling(u1 * W); A = H; B = W; }
    a1 = Math.Min(a1, A); b1 = Math.Min(b1, B);
    byte P(int across, int along) => axis == 'v' ? t.Paint[along * W + across] : t.Paint[across * W + along];
    var frac = new float[A];
    for (int a = a0; a < a1; a++)
    {
      int n = 0;
      for (int b = b0; b < b1; b++) if (P(a, b) > 0) n++;
      frac[a] = n / (float)Math.Max(1, b1 - b0);
    }
    for (int a = a0; a < a1; a++)
    {
      if (frac[a] <= 0.3f || (a > a0 && frac[a - 1] > 0.3f)) continue;
      int e = a;
      while (e < a1 && frac[e] > 0.3f) e++;
      int len = e - a;
      if (len < 2) continue;
      var painted = new bool[B];
      int ny = 0, n = 0;
      for (int b = b0; b < b1; b++)
      {
        int c = 0;
        for (int q = a; q < e; q++) { var p = P(q, b); if (p > 0) { c++; n++; if (p == 2) ny++; } }
        painted[b] = c * 2 > len;
      }
      // worn paint: gaps shorter than a twelfth of the cell are cracks, and a line painted over 70% of it is solid
      int ext = b1 - b0, gap = Math.Max(ext / 12, 2);
      for (int b = b0; b < b1; b++)
      {
        if (painted[b] || b == b0 || !painted[b - 1]) continue;
        int e2 = b;
        while (e2 < b1 && !painted[e2]) e2++;
        if (e2 < b1 && e2 - b <= gap) for (int q = b; q < e2; q++) painted[q] = true;
      }
      if (Enumerable.Range(b0, ext).Count(b => painted[b]) > 0.7f * ext) for (int b = b0; b < b1; b++) painted[b] = true;
      var band = new Band { Axis = axis, Pos = (a + len / 2f) / A, Width = len / (float)A, Colour = (byte)(ny * 2 > n ? 2 : 1), Wraps = wraps };
      band.Painted = Runs(painted);
      if (band.Painted.Count > 0) t.Bands.Add(band);
    }
  }
}
