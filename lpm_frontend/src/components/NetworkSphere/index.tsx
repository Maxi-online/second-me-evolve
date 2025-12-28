'use client';

import type React from 'react';
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

interface NameNode {
  name: string;
  size: number;
  color: string;
  position: [number, number, number];
}

interface NetworkSphereProps {
  onInitialized?: () => void;
}

const NetworkSphere: React.FC<NetworkSphereProps> = ({ onInitialized }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);

  useEffect(() => {
    if (typeof window === 'undefined') return;

    const checkWebGLAvailability = (): boolean => {
      try {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');

        return !!gl;
      } catch {
        return false;
      }
    };

    // Store animation frame ID for cleanup
    let animationFrameId: number;
    // Store scene and renderer references for cleanup
    let scene: THREE.Scene;
    let renderer: THREE.WebGLRenderer;
    let camera: THREE.PerspectiveCamera;

    // Function to initialize the sphere
    const initializeSphere = () => {
      if (!containerRef.current) return;

      // Create scene
      scene = new THREE.Scene();
      sceneRef.current = scene;
      scene.background = null; // Transparent background

      // Create camera
      camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 2000);
      camera.position.z = 650;

      // Create renderer
      try {
        if (!checkWebGLAvailability()) {
          throw new Error('Your browser does not support WebGL');
        }

        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        rendererRef.current = renderer;
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        containerRef.current.appendChild(renderer.domElement);
      } catch (error) {
        console.error('WebGL renderer creation failed:', error);

        onInitialized?.();

        return;
      }

      // Define sphere radius
      const radius = 320;

      // Create nodes
      const names: NameNode[] = [];
      const colors = ['#4ecdc4', '#ff6b6b', '#ffd93d'];
      const sizes = [0.9, 1.0, 1.1, 1.2, 1.3];

      // Create 40 nodes, reducing density
      for (let i = 0; i < 40; i++) {
        names.push({
          name: `Node${i}`,
          size: sizes[Math.floor(Math.random() * sizes.length)],
          color: colors[Math.floor(Math.random() * colors.length)],
          position: randomSpherePoint(radius)
        });
      }

      // Only create colored dots, don't display names
      const dotGeometry = new THREE.SphereGeometry(3.5, 16, 16);
      const dotMaterials = {
        '#4ecdc4': new THREE.MeshBasicMaterial({ color: 0x4ecdc4 }),
        '#ff6b6b': new THREE.MeshBasicMaterial({ color: 0xff6b6b }),
        '#ffd93d': new THREE.MeshBasicMaterial({ color: 0xffd93d })
      };

      // Create dots for each node
      names.forEach((node) => {
        const material = dotMaterials[node.color as keyof typeof dotMaterials];
        const dot = new THREE.Mesh(dotGeometry, material);

        dot.position.set(...node.position);
        scene.add(dot);
      });

      // Connect each node with more nearby nodes
      const maxConnections = 5; // Increased maximum number of connections per node
      const maxDistance = radius * 1.5; // Increased maximum distance for connections

      names.forEach((node, index) => {
        // Calculate distances to all other nodes
        const distances: { index: number; distance: number }[] = names.map(
          (otherNode, otherIndex) => {
            if (index === otherIndex)
              return {
                index: otherIndex,
                distance: Infinity
              }; // Don't connect to self

            const dx = node.position[0] - otherNode.position[0];
            const dy = node.position[1] - otherNode.position[1];
            const dz = node.position[2] - otherNode.position[2];

            return {
              index: otherIndex,
              distance: Math.sqrt(dx * dx + dy * dy + dz * dz)
            };
          }
        );

        // Sort by distance and take the closest nodes
        distances.sort((a, b) => a.distance - b.distance);
        const connectCount = Math.floor(Math.random() * maxConnections) + 1; // At least one connection

        for (let i = 0; i < connectCount && i < distances.length; i++) {
          if (distances[i].distance > maxDistance) continue;

          // Only create connection if the other node's index is higher
          // This prevents creating the same connection twice
          if (distances[i].index > index) {
            const otherNode = names[distances[i].index];

            // Create curved arc geometry that follows the sphere's surface
            const points = [];

            // Calculate the great circle arc between the two points
            // First, normalize the positions to get direction vectors
            const startPos = new THREE.Vector3(...node.position).normalize();
            const endPos = new THREE.Vector3(...otherNode.position).normalize();

            // Calculate the angle between the two points
            // const angle = startPos.angleTo(endPos);

            // Create intermediate points along the great circle arc
            const segments = 12; // More segments for smoother curves

            for (let j = 0; j <= segments; j++) {
              // Interpolation factor
              const t = j / segments;

              // Spherical linear interpolation (SLERP) between the two points
              const interpVec = new THREE.Vector3().copy(startPos).lerp(endPos, t).normalize();

              // Scale to the sphere radius
              const pointOnSphere = interpVec.multiplyScalar(radius);

              points.push(pointOnSphere);
            }

            const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);

            // Create connection with varying opacity based on distance
            const opacity = Math.max(0.25, 0.45 - (distances[i].distance / maxDistance) * 0.2);
            const lineMaterial = new THREE.LineBasicMaterial({
              color: 0xadd8e6, // Light blue color - keeping original color
              transparent: true,
              opacity: opacity
            });

            const line = new THREE.Line(lineGeometry, lineMaterial);

            scene.add(line);
          }
        }
      });

      // Animation
      const rotationSpeed = 0.0005;
      const animate = () => {
        animationFrameId = requestAnimationFrame(animate);

        // Rotate all elements
        scene.rotation.y += rotationSpeed;
        scene.rotation.x += rotationSpeed / 2;

        renderer.render(scene, camera);
      };

      animate();

      // Mark as initialized
      setIsInitialized(true);

      // Notify parent component that initialization is complete
      if (onInitialized) {
        // Use setTimeout to ensure this happens after the component is fully rendered
        setTimeout(() => {
          onInitialized();
        }, 0);
      }
    };

    // Handle window resize
    const handleResize = () => {
      if (!camera || !renderer) return;

      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };

    // Initialize the sphere
    initializeSphere();

    // Add resize event listener
    window.addEventListener('resize', handleResize);

    // Cleanup function
    return () => {
      window.removeEventListener('resize', handleResize);

      // Cancel any pending animation frames
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }

      // Remove renderer from DOM
      if (containerRef.current && rendererRef.current) {
        containerRef.current.removeChild(rendererRef.current.domElement);
      }

      // Release resources
      if (sceneRef.current) {
        sceneRef.current.clear();
      }

      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
    };
  }, []);

  // Generate random points on a sphere
  function randomSpherePoint(radius: number): [number, number, number] {
    const u = Math.random();
    const v = Math.random();
    const theta = 2 * Math.PI * u;
    const phi = Math.acos(2 * v - 1);

    const x = radius * Math.sin(phi) * Math.cos(theta);
    const y = radius * Math.sin(phi) * Math.sin(theta);
    const z = radius * Math.cos(phi);

    return [x, y, z];
  }

  // Since we no longer display names, the text sprite creation function has been removed

  return (
    <div
      ref={containerRef}
      className="fixed inset-0 -z-10 w-screen h-screen overflow-hidden"
      style={{
        pointerEvents: 'none',
        opacity: isInitialized ? 1 : 0,
        transition: 'opacity 0.5s ease-in-out'
      }}
    />
  );
};

export default NetworkSphere;
