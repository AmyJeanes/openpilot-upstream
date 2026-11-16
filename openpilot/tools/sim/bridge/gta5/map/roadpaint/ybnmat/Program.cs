// ybnmat <game> sample <out.json> [per]: every collision material's count, the area of its upward-facing triangles (ground,
//   not walls) and up to [per] spots where it lies, spread over the map.
// ybnmat <game> boxes <boxes.txt> <out.jsonl>: the upward-facing triangles inside any of the boxes ("x y r" a line), with
//   their materials: {"m": name, "p": [x1,y1,z1,...]}. Only reads the game's files.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CodeWalker.GameFiles;
using SharpDX;

var game = args[0]; var mode = args[1];
var inv = System.Globalization.CultureInfo.InvariantCulture;
GTA5Keys.LoadFromPath(game, true, null);
var rpf = new RpfManager();
rpf.Init(game, true, s => { }, e => Console.Error.WriteLine(e));
var mats = new List<string>();
var txt = rpf.GetFileUTF8Text(@"update\update.rpf\common\data\materials\materials.dat") ?? rpf.GetFileUTF8Text(@"common.rpf\data\materials\materials.dat");
foreach (var line in txt.Split('\n'))
{
  var l = line.Trim();
  if (l.Length == 0 || l[0] == '#' || l.StartsWith("Version", StringComparison.OrdinalIgnoreCase)) continue;
  var name = l.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries)[0];
  if (char.IsDigit(name[0])) continue;
  mats.Add(name);
}
var ybns = new Dictionary<string, RpfFileEntry>();
foreach (var r in rpf.AllRpfs)
{
  var lp = r.Path.ToLowerInvariant();
  if (!(lp.StartsWith("x64") || lp.StartsWith("update\\"))) continue;
  foreach (var e in r.AllEntries)
    if (e is RpfFileEntry fe && e.NameLower.EndsWith(".ybn") && !e.NameLower.StartsWith("hi@")) ybns[e.NameLower] = fe;
}
var boxes = new List<(float x, float y, float r)>();
StreamWriter w = null;
int per = 8;
if (mode == "boxes")
{
  foreach (var l in File.ReadAllLines(args[2])) { var q = l.Split(' ', StringSplitOptions.RemoveEmptyEntries); if (q.Length >= 3) boxes.Add((float.Parse(q[0], inv), float.Parse(q[1], inv), float.Parse(q[2], inv))); }
  w = new StreamWriter(args[3]);
}
else if (args.Length > 3) per = int.Parse(args[3]);
var count = new Dictionary<string, long>(); var area = new Dictionary<string, double>();
var spots = new Dictionary<string, List<(float x, float y, float z)>>(); var seen = new Dictionary<string, long>();
var rnd = new Random(7);
foreach (var (name, fe) in ybns)
{
  YbnFile y;
  try { y = rpf.GetFile<YbnFile>(fe); } catch { continue; }
  var b = y?.Bounds;
  if (b == null) continue;
  if (mode == "boxes" && !boxes.Any(q => !(b.BoxMax.X < q.x - q.r || b.BoxMin.X > q.x + q.r || b.BoxMax.Y < q.y - q.r || b.BoxMin.Y > q.y + q.r))) continue;
  var geoms = new List<BoundGeometry>();
  void Walk(Bounds bb) { if (bb is BoundComposite c) { if (c.Children?.data_items != null) foreach (var ch in c.Children.data_items) if (ch != null) Walk(ch); } else if (bb is BoundGeometry g) geoms.Add(g); }
  Walk(b);
  foreach (var g in geoms)
  {
    if (g.Polygons == null) continue;
    for (int i = 0; i < g.Polygons.Length; i++)
    {
      if (g.Polygons[i] is not BoundPolygonTriangle t) continue;
      var p1 = t.Vertex1; var p2 = t.Vertex2; var p3 = t.Vertex3;
      var n = Vector3.Cross(p2 - p1, p3 - p1); var a2 = n.Length();
      if (a2 < 1e-6f || Math.Abs(n.Z) / a2 < 0.7f) continue;  // walls and steep faces: not ground
      var cx = (p1.X + p2.X + p3.X) / 3; var cy = (p1.Y + p2.Y + p3.Y) / 3;
      var mi = g.GetMaterial(i).Type.Index;
      var mn = mi < mats.Count ? mats[mi] : $"#{mi}";
      if (mode == "boxes")
      {
        if (!boxes.Any(q => Math.Abs(cx - q.x) <= q.r && Math.Abs(cy - q.y) <= q.r)) continue;
        w.WriteLine(string.Format(inv, "{{\"m\":\"{0}\",\"p\":[{1:F2},{2:F2},{3:F2},{4:F2},{5:F2},{6:F2},{7:F2},{8:F2},{9:F2}]}}", mn, p1.X, p1.Y, p1.Z, p2.X, p2.Y, p2.Z, p3.X, p3.Y, p3.Z));
        continue;
      }
      count[mn] = count.GetValueOrDefault(mn) + 1; area[mn] = area.GetValueOrDefault(mn) + a2 / 2;
      if (a2 / 2 < 4 || Math.Abs(cx) > 4000 || cy < -3400 || cy > 6900) continue;  // a spot worth a picture: a fair-sized face, inside the pictured map
      if (!spots.TryGetValue(mn, out var L)) spots[mn] = L = new List<(float, float, float)>();
      var k = seen[mn] = seen.GetValueOrDefault(mn) + 1;
      if (L.Count < per) L.Add((cx, cy, (p1.Z + p2.Z + p3.Z) / 3));
      else { var j = (long)(rnd.NextDouble() * k); if (j < per) L[(int)j] = (cx, cy, (p1.Z + p2.Z + p3.Z) / 3); }
    }
  }
}
if (mode == "boxes") { w.Dispose(); return 0; }
using (var o = new StreamWriter(args[2]))
{
  o.WriteLine("[");
  var first = true;
  foreach (var k in area.OrderByDescending(k => k.Value))
  {
    var sp = spots.TryGetValue(k.Key, out var L) ? string.Join(",", L.Select(s => string.Format(inv, "[{0:F1},{1:F1},{2:F1}]", s.x, s.y, s.z))) : "";
    o.WriteLine((first ? "" : ",") + string.Format(inv, "{{\"m\":\"{0}\",\"n\":{1},\"km2\":{2:F4},\"spots\":[{3}]}}", k.Key, count[k.Key], k.Value / 1e6, sp));
    first = false;
  }
  o.WriteLine("]");
}
return 0;
