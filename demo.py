"""Small deterministic demonstration using synthetic inputs only."""
import json
from pathlib import Path
import numpy as np
from topology import diagram_for, landscape_norms


def main():
    square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    diagram = diagram_for(square)
    np.testing.assert_allclose(diagram, [[1., np.sqrt(2)]], rtol=1e-6)
    rng = np.random.default_rng(20260928)
    points = rng.normal(size=(50, 4))
    base = landscape_norms(diagram_for(points))[0]
    scaled = landscape_norms(diagram_for(3 * points))[0]
    reversed_sign = landscape_norms(diagram_for(-points))[0]
    np.testing.assert_allclose(scaled, 9 * base, rtol=1e-5)
    np.testing.assert_allclose(reversed_sign, base, rtol=1e-6)
    result = {
        'input': 'square and synthetic Gaussian points; no market data',
        'square_h1_diagram': diagram.tolist(),
        'synthetic_l1': base,
        'scale_x3_l1_ratio': scaled / base,
        'sign_reversal_l1_ratio': reversed_sign / base,
        'interpretation': 'L1 changes with squared common scale; whole-cloud sign reversal is invisible.',
    }
    out = Path(__file__).resolve().parent / 'demo_output'
    out.mkdir(exist_ok=True)
    (out / 'synthetic_summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
