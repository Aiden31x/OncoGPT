import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import SideBar from "@/Components/SideBar";
import { SessionProvider } from "next-auth/react";
import { Providers } from "./providers";
import { getServerSession } from "next-auth";
import { authOptions } from "./api/auth/[...nextauth]/route";
import Login from "@/Components/Login";
import ClientProvider from "@/Components/ClientProvider";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Create Next App",
  description: "Generated by create next app",
};

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const session = await getServerSession(authOptions);

  console.log(session);
  return (
    <html lang="en">
      <head></head>
      <body className={inter.className}>
        <Providers session={session}>
          {!session ? (
            <Login></Login>
          ) : (
            <div className="flex">
              <div className="bg-slate-200 mx-w-xs h-screen overflow-y-auto md:min-w-[16rem]">
                <SideBar />
              </div>

              <ClientProvider/>
              
              <div className="flex-1 bg-black">{children}</div>
            </div>
          )}
        </Providers>
      </body>
    </html>
  );
}
