export default function Navbar() {
  return (
    <nav className="absolute max-w-7xl mx-auto left-0 right-0 top-0 w-full z-50 px-12 py-10 flex justify-start items-center bg-transparent">
      <div className="flex items-center gap-3">
        <img src="/logo.png" alt="Logo" className="w-18 h-auto" />
        <span className="text-white font-bold text-3xl tracking-tight">
          Bind Hard
        </span>
      </div>
    </nav>
  );
}
