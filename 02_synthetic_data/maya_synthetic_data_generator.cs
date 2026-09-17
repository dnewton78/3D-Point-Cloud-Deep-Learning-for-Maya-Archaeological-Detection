
  private void RunScript(int dataNum, double size, int divisions, double maxHeight, double minHeight, double stdDevMax, double stdDevMin, int maxStructure, int minStructure, double maxStructSize, double minStructSize, double lineSquareBalance, int hillNum, double hillMaxH, double hillMinH, double hillMaxW, double hillMinW, double hillMaxL, double hillMinL, double noiseMean, double noiseSdv, bool CSVExport, string filePath, string fileName, out object points, out object classifications, out object slopes, out object A, out object B)
  {


    //1 Create base topography
    //create nurbs surface
    //move points randomly
    //output surfaces
    //2 Create mounds
    // Rebuild Surfaces
    // select subset of points and lift
    // add noise level
    // output point list as CVV
    //generate base point grid
    Random rGen = new Random(System.DateTime.Now.Millisecond);
    List < Point3d[,] > dataSetP = new List<Point3d[,]>();
    List < double[,] > dataSetS = new List<double[,]>();
    List < int[,] > dataSetC = new List<int[,]>();
    List < Line > tlList = new List < Line >();
    List < Line > lList = new List < Line >();

    for(int i = 0; i < dataNum; i++)
    {
      int ptNum = divisions + 1;//add extra point to account for point at 0 and end of length
      Point3d [,] pList = new Point3d[ptNum, ptNum];
      double [,] sList = new double[ptNum, ptNum];
      int [,] cList = new int[ptNum, ptNum];

      //generate point grid size
      generatePointGrid(size, size, ptNum, ref pList, ref cList, ref sList);

      Point3d maxP = pList[ptNum - 1, ptNum - 1];
      Point3d minP = pList[0, 0];

      //generate topography
      tlList = generateTopo(hillNum, hillMaxL, hillMinL, maxP, minP, rGen);

      //move points for hills
      double height = GetRandomDouble(rGen, hillMinH, hillMaxH);
      double stdDev = GetRandomDouble(rGen, hillMinW, hillMaxW);
      movePointsWithGaussian(height, 0, stdDev, true, stdDev * 3, tlList, ref pList, ref cList);

      //generate structures
      int sNum = (int) GetRandomDouble(rGen, (double) minStructure, (double) maxStructure);
      List < Line > slList = new List < Line >();
      for(int j = 0; j < sNum; j++)
      {
        slList.Add(tlList[rGen.Next(0, tlList.Count)]);
      }
      lList = generateStructures(lineSquareBalance, minStructSize, maxStructSize, maxP, minP, rGen, slList);

      //move points for structures
      height = GetRandomDouble(rGen, minHeight, maxHeight);
      stdDev = GetRandomDouble(rGen, stdDevMin, stdDevMax);
      movePointsWithGaussian(height, 0, stdDev, false, stdDev * 2, lList, ref pList, ref cList);

      //add noise
      AddGaussianNoise(noiseMean, noiseSdv, ref pList, rGen);

      //calculate slopes
      calcSlope(ref pList, ref sList);

      dataSetP.Add(pList);
      dataSetS.Add(sList);
      dataSetC.Add(cList);
    }


    if(CSVExport)
      WriteToCsv(ref dataSetP, ref dataSetS, ref dataSetC, fileName, filePath);


    points = dataSetP[0];
    classifications = dataSetC[0];
    slopes = dataSetS[0];
    A = tlList;
    B = lList;
  }

  // <Custom additional code> 

  private double GenerateGaussianNoise(double mean, double stdDev, Random random)
  {
    double u1 = 1.0 - random.NextDouble(); // uniform(0,1] random doubles
    double u2 = 1.0 - random.NextDouble();
    double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
      Math.Sin(2.0 * Math.PI * u2); // random normal(0,1)
    return mean + stdDev * randStdNormal; // random normal(mean,stdDev^2)
  }

  public void AddGaussianNoise(double mean, double stdDev, ref Point3d[,] pList, Random random)
  {
    for(int i = 0; i < pList.GetLength(0); i++)
    {
      for(int j = 0; j < pList.GetLength(1); j++)
      {
        Point3d temp = pList[i, j];
        double noiseX = GenerateGaussianNoise(mean, stdDev, random);
        double noiseY = GenerateGaussianNoise(mean, stdDev, random);
        double noiseZ = GenerateGaussianNoise(mean, stdDev, random);
        temp.X = temp.X + noiseX;
        temp.Y = temp.Y + noiseY;
        temp.Z = temp.Z + noiseZ;
        pList[i, j] = temp;
      }
    }
  }

  public void WriteToCsv(ref List <Point3d[,] > dataSetP, ref List < double[,] > dataSetS, ref List < int[,] > dataSetC, string fileName, string filePath)
  {
    for (int i = 0; i < dataSetP.Count; i++)
    {
      string path = filePath + "\\" + i + fileName + ".csv";
      WritePointsToCsv(dataSetP[i], dataSetS[i], dataSetC[i], path);
    }
  }

  public void WritePointsToCsv(Point3d[,] points, double[,] slopes, int[,] classes, string path)
  {
    using (System.IO.StreamWriter file = new System.IO.StreamWriter(@path,false))
    {
      //P1 starting position
      //file.WriteLine();
      // System.Text.StringBuilder csvContent = new StringBuilder();

      int rows = points.GetLength(0);
      int columns = points.GetLength(1);

      for (int i = 0; i < rows; i++)
      {
        for (int j = 0; j < columns; j++)
        {
          Point3d point = points[i, j];
          double slope = slopes[i, j];
          int cl = classes[i, j];
          if (point != null)
          {
            // Create a CSV line for the current point
            string line = Math.Round(point.X,3) + "," + Math.Round(point.Y,3) + "," + Math.Round(point.Z,3) + "," + Math.Round(slope,3) + "," + cl;
            file.WriteLine(line);
          }
        }
      }
    }

    // Write the CSV content to a file
    //File.WriteAllText(filePath, csvContent.ToString());
  }

  double GetRandomDouble(Random random, double min, double max)
  {
    return min + (random.NextDouble() * (max - min));
  }

  List<Line> generateTopo(int hillNum, double maxL, double minL, Point3d maxP, Point3d minP, Random rGen)
  {
    List<Line> lineList = new List<Line>();

    for(int i = 0; i < hillNum; i++)
    {

      //create a line and add it to the list
      double xmin = minP.X;
      double xmax = maxP.X;
      double ymin = minP.Y;
      double ymax = maxP.Y;

      //pick random point 1
      double randX = GetRandomDouble(rGen, xmin, xmax);
      double randY = GetRandomDouble(rGen, ymin, ymax);

      //pick random point 2
      double length = GetRandomDouble(rGen, minL, maxL);
      double randX2 = GetRandomDouble(rGen, randX - length, randX + length);
      double randY2 = GetRandomDouble(rGen, randY - length, randY + length);

      //create points
      Point3d p1 = new Point3d(randX, randY, 0);
      Point3d p2 = new Point3d(randX2, randY2, 0);

      //create line
      Line line = new Line(p1, p2);
      lineList.Add(line);

    }

    return lineList;

  }

  List<Line> generateStructures(double percentLine, double minSize, double maxSize, Point3d maxP, Point3d minP, Random rGen, List<Line> slList)
  {
    List<Line> lineList = new List<Line>();

    for(int i = 0; i < slList.Count; i++)
    {
      Point3d cP = slList[i].PointAtLength(slList[i].Length / 2);

      if(rGen.NextDouble() <= percentLine)
      {
        //pick random point 2
        double length = GetRandomDouble(rGen, minSize, maxSize);
        double randX2 = GetRandomDouble(rGen, cP.X - length / 2, cP.X + length / 2);
        double randY2 = GetRandomDouble(rGen, cP.Y - length / 2, cP.Y + length / 2);

        //create points
        Point3d p2 = new Point3d(randX2, randY2, 0);
        Vector3d v1 = Point3d.Subtract(cP, p2);
        Point3d p3 = Point3d.Add(cP, v1);
        //create line
        Line line = new Line(p3, p2);
        lineList.Add(line);
      }
      else
      {
        //create rectangle
        double width = GetRandomDouble(rGen, minSize, maxSize);
        double length = GetRandomDouble(rGen, minSize, maxSize);
        double angle = GetRandomDouble(rGen, 0, Math.PI * 2);
        Rectangle3d rect = new Rectangle3d(new Plane(cP, Vector3d.ZAxis), new Interval(-width / 2, width / 2), new Interval(-length / 2, length / 2));
        rect.Transform(Transform.Rotation(angle, cP));

        //create line
        Line l1 = new Line(rect.Corner(0), rect.Corner(1));
        Line l2 = new Line(rect.Corner(1), rect.Corner(2));
        Line l3 = new Line(rect.Corner(2), rect.Corner(3));
        Line l4 = new Line(rect.Corner(3), rect.Corner(0));
        lineList.Add(l1);
        lineList.Add(l2);
        lineList.Add(l3);
        lineList.Add(l4);
      }
    }

    return lineList;
  }

  List<Line> generateStructures(int sNumber, double percentLine, double minSize, double maxSize, Point3d maxP, Point3d minP, Random rGen)
  {
    List<Line> lineList = new List<Line>();

    for(int i = 0; i < sNumber; i++)
    {
      if(rGen.NextDouble() <= percentLine)
      {
        //create a line and add it to the list
        double xmin = minP.X;
        double xmax = maxP.X;
        double ymin = minP.Y;
        double ymax = maxP.Y;

        //pick random point 1
        double randX = GetRandomDouble(rGen, xmin, xmax);
        double randY = GetRandomDouble(rGen, ymin, ymax);

        //pick random point 2
        double length = GetRandomDouble(rGen, minSize, maxSize);
        double randX2 = GetRandomDouble(rGen, randX - length, randX + length);
        double randY2 = GetRandomDouble(rGen, randY - length, randY + length);

        //create points
        Point3d p1 = new Point3d(randX, randY, 0);
        Point3d p2 = new Point3d(randX2, randY2, 0);

        //create line
        Line line = new Line(p1, p2);
        lineList.Add(line);
      }
      else
      {
        double xmin = minP.X;
        double xmax = maxP.X;
        double ymin = minP.Y;
        double ymax = maxP.Y;

        //pick random point 1
        double randX = GetRandomDouble(rGen, xmin, xmax);
        double randY = GetRandomDouble(rGen, ymin, ymax);
        //create points
        Point3d p1 = new Point3d(randX, randY, 0);

        //create rectangle
        double width = GetRandomDouble(rGen, minSize, maxSize);
        double length = GetRandomDouble(rGen, minSize, maxSize);
        double angle = GetRandomDouble(rGen, 0, Math.PI * 2);
        Rectangle3d rect = new Rectangle3d(new Plane(p1, Vector3d.ZAxis), width, length);
        rect.Transform(Transform.Rotation(angle, p1));

        //create line
        Line l1 = new Line(rect.Corner(0), rect.Corner(1));
        Line l2 = new Line(rect.Corner(1), rect.Corner(2));
        Line l3 = new Line(rect.Corner(2), rect.Corner(3));
        Line l4 = new Line(rect.Corner(3), rect.Corner(0));
        lineList.Add(l1);
        lineList.Add(l2);
        lineList.Add(l3);
        lineList.Add(l4);
      }
    }

    return lineList;
  }

  void movePointsWithGaussian(double height, double center, double stdDev, bool isTopo, double buffer, List<Line> lList, ref Point3d [,] pList, ref int [,] cList)
  {
    for(int h = 0; h < lList.Count; h++)
    {
      Line tempL = lList[h];
      Point3d start = tempL.PointAt(0);
      Point3d end = tempL.PointAtLength(tempL.Length);

      for(int i = 0; i < pList.GetLength(0); i++)
      {
        for(int j = 0; j < pList.GetLength(1); j++)
        {
          Point3d temp = pList[i, j];
          //set z compnent to 0 so ditance in x-y plane is only distance
          double tempZ = temp.Z;
          temp.Z = 0;
          //get distance in z direction to move
          double distance = distanceLineToPoint(start, end, temp);
          if(distance <= buffer && cList[i, j] == 2)
          {
            double zDist = height * Math.Exp(-Math.Pow((distance - center), 2) / (2 * Math.Pow(stdDev, 2)));
            //add height to current z value
            temp.Z = zDist;

            if(!isTopo) //set classification if for structure and not topt creation
              cList[i, j] = 6;
          }
          temp.Z = temp.Z + tempZ;
          pList[i, j] = temp;
        }
      }
    }
  }


  double distanceLineToPoint(Point3d start, Point3d end, Point3d pt)
  {
    double distance = 0;

    //construct a vector
    Vector3d lineVec = Point3d.Subtract(end, start);
    double lineVecL = lineVec.Length;
    Vector3d lineVecC = new Vector3d(lineVec);
    Vector3d ptVec = Point3d.Subtract(pt, start);
    double aDotb = Vector3d.Multiply(lineVec, ptVec);
    double pVecLength = aDotb / (lineVec.Length);

    //create projected vectors
    lineVec.Unitize();
    Vector3d pVec = Vector3d.Multiply(lineVec, pVecLength);
    Vector3d cVec = Vector3d.Subtract(ptVec, pVec);
    distance = cVec.Length;

    //if projected vector is greater then length between roinal pints or a negative value then calulate the distnce to the start or endpoint and use for gaussian

    if(pVecLength >= lineVecL && aDotb > 0)//measure to end point
    {
      Vector3d tv = Point3d.Subtract(pt, end);
      distance = tv.Length;
    } else if(aDotb < 0)//measure to start point
    {
      Vector3d tv = Point3d.Subtract(pt, start);
      distance = tv.Length;
    }

    return distance;
  }

  public void generatePointGrid(double xSize, double ySize, int ptNum, ref Point3d [,] pList, ref int [,] cList, ref double [,] sList)
  {
    //point incrment
    double xIncr = xSize / ((double) ptNum);
    double yIncr = ySize / ((double) ptNum);


    for (int i = 0;i < ptNum;i++)
    {
      for (int j = 0;j < ptNum;j++)
      {

        Point3d p = new Point3d(xIncr * i, yIncr * j, 0);
        pList[i, j] = p;
        cList[i, j] = 2;
      }
    }
  }

  public void calcSlope(ref Point3d [,] pList, ref double [,] sList)
  {
    for (int i = 0;i < pList.GetLength(0);i++)
    {
      for (int j = 0;j < pList.GetLength(1);j++)
      {

        double slope = 0;
        double count = 0;
        double run = 0;
        double hieght = 0;

        //w i-1, j
        if(i - 1 >= 0)
        {
          Point3d pt = new Point3d(pList[i, j].X, pList[i, j].Y, 0);
          Point3d ptN = new Point3d(pList[i - 1, j].X, pList[i - 1, j].Y, 0);
          double distance = ptN.DistanceTo(pt);
          count++;
          run = distance;
          hieght = pList[i, j].Z - pList[i - 1, j].Z;
          slope = slope + Math.Abs((hieght / run));
        }
        //nw i-1, j+1
        if(i - 1 >= 0 && j + 1 < pList.GetLength(1))
        {
          Point3d pt = new Point3d(pList[i, j].X, pList[i, j].Y, 0);
          Point3d ptN = new Point3d(pList[i - 1, j + 1].X, pList[i - 1, j + 1].Y, 0);
          double distance = ptN.DistanceTo(pt);
          count++;
          run = distance;
          hieght = pList[i, j].Z - pList[i - 1, j + 1].Z;
          slope = slope + Math.Abs((hieght / run));
        }
        //n i, j+1
        if(j + 1 < pList.GetLength(1))
        {
          Point3d pt = new Point3d(pList[i, j].X, pList[i, j].Y, 0);
          Point3d ptN = new Point3d(pList[i, j + 1].X, pList[i, j + 1].Y, 0);
          double distance = ptN.DistanceTo(pt);
          count++;
          run = distance;
          hieght = pList[i, j].Z - pList[i, j + 1].Z;
          slope = slope + Math.Abs((hieght / run));
        }
        //ne i+1, j+1
        if(i + 1 < pList.GetLength(0) && j + 1 < pList.GetLength(1))
        {
          Point3d pt = new Point3d(pList[i, j].X, pList[i, j].Y, 0);
          Point3d ptN = new Point3d(pList[i + 1, j + 1].X, pList[i + 1, j + 1].Y, 0);
          double distance = ptN.DistanceTo(pt);
          count++;
          run = distance;
          hieght = pList[i, j].Z - pList[i + 1, j + 1].Z;
          slope = slope + Math.Abs((hieght / run));
        }
        //e i+1, j
        if(i + 1 < pList.GetLength(0))
        {
          Point3d pt = new Point3d(pList[i, j].X, pList[i, j].Y, 0);
          Point3d ptN = new Point3d(pList[i + 1, j].X, pList[i + 1, j].Y, 0);
          double distance = ptN.DistanceTo(pt);
          count++;
          run = distance;
          hieght = pList[i, j].Z - pList[i + 1, j].Z;
          slope = slope + Math.Abs((hieght / run));
        }
        //se i+1, j-1
        if(i + 1 < pList.GetLength(0) && j - 1 >= 0)
        {
          Point3d pt = new Point3d(pList[i, j].X, pList[i, j].Y, 0);
          Point3d ptN = new Point3d(pList[i + 1, j - 1].X, pList[i + 1, j - 1].Y, 0);
          double distance = ptN.DistanceTo(pt);
          count++;
          run = distance;
          hieght = pList[i, j].Z - pList[i + 1, j - 1].Z;
          slope = slope + Math.Abs((hieght / run));
        }
        //s i, j-1
        if(j - 1 >= 0)
        {
          Point3d pt = new Point3d(pList[i, j].X, pList[i, j].Y, 0);
          Point3d ptN = new Point3d(pList[i, j - 1].X, pList[i, j - 1].Y, 0);
          double distance = ptN.DistanceTo(pt);
          count++;
          run = distance;
          hieght = pList[i, j].Z - pList[i, j - 1].Z;
          slope = slope + Math.Abs((hieght / run));
        }
        //sw i-1, j-1
        if(i - 1 >= 0 && j - 1 >= 0)
        {
          Point3d pt = new Point3d(pList[i, j].X, pList[i, j].Y, 0);
          Point3d ptN = new Point3d(pList[i - 1, j - 1].X, pList[i - 1, j - 1].Y, 0);
          double distance = ptN.DistanceTo(pt);
          count++;
          run = distance;
          hieght = pList[i, j].Z - pList[i - 1, j - 1].Z;
          slope = slope + Math.Abs((hieght / run));
        }
        sList[i, j] = Math.Round((Math.Atan(slope / count)) * (180 / Math.PI), 3);
      }
    }
  }

  // </Custom additional code> 
}
