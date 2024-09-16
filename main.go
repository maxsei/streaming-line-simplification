package main

import (
	"container/heap"
	"fmt"
	"image/color"
	"math"
	"math/rand"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

var Inf32 = float32(math.Inf(1))

func main() {
	r := rand.New(rand.NewSource(0))
	// Random walk.
	actual := make([][]float32, 3*100)
	{
		prev := float32(0)
		for i := range actual {
			// v := []float32{float32(i), float32(i)}
			v := []float32{float32(i), prev}
			prev += r.Float32()*2 - 1
			actual[i] = v
		}
	}

	// Simplify.
	s := NewSimplification(100)
	for _, v := range actual {
		s.Update(v)
	}
	simp := s.Get()

	// Plot.
	fmt.Printf("len(actual): %d len(simp): %d\n", len(actual), len(simp))
	plotPoints(F32XYs(actual), F32XYs(simp), "out.png")
}

// TODO: should probably providethe simplification with an order function
// (currently the order is based on the order they are updated in) and a
// distance function.
type (
	Node struct {
		point []float32
		prev  *Node
		next  *Node
		q     float32
		index int
	}
	Simplification struct {
		pq   QNodePQ
		head *Node
		tail *Node
	}
)

func hausdorff(pp1, pp2 [][]float32) float32 {
	bg, sm := pp1, pp2
	if len(pp1) < len(pp2) {
		bg, sm = pp2, pp1
	}
	res := -Inf32
	for _, p1 := range bg {
		shortest := Inf32
		for _, p2 := range sm {
			d := dist(p1, p2)
			if d < shortest {
				shortest = d
			}
		}
		if shortest > res {
			res = shortest
		}
	}
	return res
}
func dist(p1, p2 []float32) float32 {
	var sum float32
	for i := range p1 {
		diff := p2[i] - p1[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

func (n *Node) calcError() float32 {
	l := n.prev
	r := n.next
	if l == nil || r == nil {
		return Inf32
	}
	path := [...][]float32{l.point, n.point, r.point}
	segment := [...][]float32{l.point, r.point}
	return hausdorff(path[:], segment[:])
	// return FrechetDistance(path[:], segment[:])
}

// plotPoints takes two sets of points and plots them on the same plot,
// coloring the first set red and the second set blue.
func plotPoints(points1, points2 plotter.XYer, outputFilename string) {
	p := plot.New()
	p.Title.Text = "Two Sets of Points"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	line1, err := plotter.NewLine(points1)
	if err != nil {
		panic(err)
	}
	line1.Color = color.RGBA{R: 255, A: 255}

	line2, err := plotter.NewLine(points2)
	if err != nil {
		panic(err)
	}
	line2.Color = color.RGBA{B: 255, A: 255}

	p.Add(line1, line2)

	if err := p.Save(14*vg.Inch, 6*vg.Inch, outputFilename); err != nil {
		panic(err)
	}
}

type F32XYs [][]float32

func (xy F32XYs) Len() int                { return len(xy) }
func (xy F32XYs) XY(i int) (x, y float64) { return float64(xy[i][0]), float64(xy[i][1]) }

func NewSimplification(l int) *Simplification {
	return &Simplification{
		pq: make(QNodePQ, 0, l-1),
	}
}

type QNodePQ []*Node

var _ heap.Interface = new(QNodePQ)

func (pq *QNodePQ) Push(x any) {
	n := x.(*Node)
	n.index = len(*pq)
	*pq = append(*pq, n)
}
func (pq *QNodePQ) Pop() any {
	if len(*pq) == 0 {
		return nil
	}
	res := (*pq)[len(*pq)-1]
	*pq = (*pq)[:len(*pq)-1]
	return res
}

func (pq QNodePQ) Len() int { return len(pq) }

func (pq QNodePQ) Less(i, j int) bool { return pq[i].q < pq[j].q }

func (pq QNodePQ) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (s *Simplification) Update(point []float32) {
	// Create new node
	n := new(Node)
	n.point = point
	n.q = Inf32

	// Append to linked list.
	if s.head == nil {
		s.head = n
		return
	}
	if s.tail == nil {
		s.tail = n
		s.tail.prev = s.head
		s.head.next = s.tail
	} else {
		s.tail.next = n
		n.prev = s.tail
		s.tail = n
	}

	// Update previous error value.
	if prev := n.prev; prev != nil {
		q := prev.calcError()
		prev.q = q
	}

	// Build up priority queue.
	if len(s.pq) < cap(s.pq) {
		s.pq = append(s.pq, n)
		if len(s.pq) == cap(s.pq)-1 {
			heap.Init(&s.pq)
		}
		return
	}

	// XXX: This could be optimized by checking if it was just inserted
	removed := heap.Pop(&s.pq).(*Node)
	heap.Push(&s.pq, n)

	switch removed {
	case s.tail:
		// fmt.Println("removed tail")
		s.tail = removed.prev
	case s.head:
		fmt.Println("removed head")
		fmt.Println()
		s.head = removed.next
	default:
		// fmt.Println("removed node")
	}

	if prev := removed.prev; prev != nil {
		prev.next = removed.next
		prev.q = prev.calcError()
		heap.Fix(&s.pq, prev.index)
	}
	if next := removed.next; next != nil {
		next.prev = removed.prev
		next.q = next.calcError()
		heap.Fix(&s.pq, next.index)
	}
}

func (s *Simplification) Get() [][]float32 {
	res := make([][]float32, 0, len(s.pq)+1)
	for cur := s.head; cur != nil; cur = cur.next {
		res = append(res, cur.point)
	}
	return res
}

// FrechetDistance calculates the Fréchet distance between two sequences of points.
// It uses dynamic programming to efficiently compute the distance.
func FrechetDistance(P, Q [][]float32) float32 {
	cache := make([][]float32, len(P))
	for i := range cache {
		cache[i] = make([]float32, len(Q))
		for j := range cache[i] {
			cache[i][j] = -1.0
		}
	}
	return frechet(P, Q, len(P)-1, len(Q)-1, cache)
}

// frechet is a recursive helper function that computes the Fréchet distance.
func frechet(P, Q [][]float32, i, j int, cache [][]float32) float32 {
	if cache[i][j] != -1.0 {
		return cache[i][j]
	}

	// Base case: both curves have a single point
	if i == 0 && j == 0 {
		cache[i][j] = dist(P[0], Q[0])
		return cache[i][j]
	}

	// Base case: first curve has a single point
	if i == 0 {
		f := frechet(P, Q, 0, j-1, cache)
		d := dist(P[0], Q[j])
		cache[i][j] = float32(math.Max(float64(f), float64(d)))
		return cache[i][j]
	}

	// Base case: second curve has a single point
	if j == 0 {
		f := frechet(P, Q, i-1, 0, cache)
		d := dist(P[i], Q[0])
		cache[i][j] = float32(math.Max(float64(f), float64(d)))
		return cache[i][j]
	}

	// Recurrence relation
	minDist := math.Min(
		math.Min(
			float64(frechet(P, Q, i-1, j, cache)),
			float64(frechet(P, Q, i, j-1, cache)),
		),
		float64(frechet(P, Q, i-1, j-1, cache)),
	)

	cache[i][j] = float32(math.Max(minDist, float64(dist(P[i], Q[j]))))
	return cache[i][j]
}
