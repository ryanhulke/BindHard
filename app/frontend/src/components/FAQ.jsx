import { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { PiStarFourBold } from "react-icons/pi";

const FAQ_DATA = [
  {
    q: "What input data does Bind Hard need?",
    a: "A single PDB file containing your target protein pocket's structure. Upload it through the dashboard and the platform will match it against precomputed binding trajectories to identify candidate binders and their predicted 3D conformations.",
  },
  {
    q: "How does the binding prediction work?",
    a: "Bind Hard uses a flow matching model built on an equivariant graph neural network (EGNN). Given a target protein, the model generates candidate small-molecule binders by learning the distribution of binding conformations from a large dataset of known protein-ligand complexes.",
  },
  {
    q: "What does the trajectory viewer show?",
    a: "It renders the full binding trajectory — the sequence of 3D conformations the candidate binder explores as it docks into the target's binding site. You can scrub frame-by-frame or play it as an animation.",
  },
  {
    q: "How are candidates ranked?",
    a: "Candidate binders are sorted by their Vina docking score (lower is better), which estimates binding affinity in kcal/mol.",
  },
  {
    q: "Can I use this for drug discovery?",
    a: "Bind Hard is designed to accelerate the early-stage hit identification step. It helps researchers quickly explore which molecular scaffolds might bind a given target and visualize how they fit in the binding pocket. Hits should be followed up with wet-lab validation and more detailed simulations before drawing conclusions about clinical relevance.",
  },
];

function FaqItem({ q, a, index }) {
  const [open, setOpen] = useState(false);
  const contentRef = useRef(null);
  const itemRef = useRef(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = itemRef.current;
    if (!el) return;
    const ob = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setVisible(true);
          ob.disconnect();
        }
      },
      { threshold: 0.1 },
    );
    ob.observe(el);
    return () => ob.disconnect();
  }, []);

  return (
    <div
      ref={itemRef}
      className="border-b border-white/8 transition-all duration-700"
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0)" : "translateY(20px)",
        transitionDelay: `${index * 60}ms`,
      }}
    >
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between py-6 text-left group cursor-pointer"
      >
        <span className="text-white font-semibold text-base md:text-lg pr-8 group-hover:text-white transition-colors">
          {q}
        </span>
        <span
          className="text-white text-2xl transition-transform duration-300 shrink-0"
          style={{ transform: open ? "rotate(45deg)" : "rotate(0)" }}
        >
          +
        </span>
      </button>
      <div
        className="overflow-hidden transition-all duration-400 ease-in-out"
        style={{
          maxHeight: open
            ? (contentRef.current?.scrollHeight ?? 500) + "px"
            : "0px",
        }}
      >
        <p
          ref={contentRef}
          className="text-white text-sm leading-relaxed pb-6 pr-12"
        >
          {a}
        </p>
      </div>
    </div>
  );
}

export default function FAQ() {
  const navigate = useNavigate();

  return (
    <section
      id="faq"
      className="relative z-10 w-full max-w-4xl mx-auto px-6 pt-40 pb-32"
    >
      {/* Section heading */}
      <div className="text-center mb-16">
        <p className="text-white text-xs font-bold tracking-[0.3em] uppercase mb-4">
          Questions
        </p>
        <h2 className="font-eb-garamond text-white text-5xl md:text-7xl font-bold tracking-tight leading-[1.05]">
          FAQ
        </h2>
      </div>

      {/* FAQ list */}
      <div>
        {FAQ_DATA.map((item, i) => (
          <FaqItem key={i} q={item.q} a={item.a} index={i} />
        ))}
      </div>
    </section>
  );
}
