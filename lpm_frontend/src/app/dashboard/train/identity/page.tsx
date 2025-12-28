'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import type { ILoadInfo } from '@/service/info';
import { updateLoadInfo } from '@/service/info';
import { useLoadInfoStore } from '@/store/useLoadInfoStore';
import { ROUTER_PATH } from '@/utils/router';
import { message } from 'antd';

export default function IdentityPage() {
  const pageTitle = 'Define Your Identity';
  const pageDescription = "Build your AI's foundation with your basic information.";

  const router = useRouter();
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [email, setEmail] = useState('');
  const [originalName, setOriginalName] = useState('');
  const [originalDescription, setOriginalDescription] = useState('');
  const [originalEmail, setOriginalEmail] = useState('');
  const [isEdited, setIsEdited] = useState(false);
  const [emailError, setEmailError] = useState('');
  const loadInfo = useLoadInfoStore((state) => state.loadInfo);
  const fetchLoadInfo = useLoadInfoStore((state) => state.fetchLoadInfo);

  const setInfo = (localInfo: ILoadInfo) => {
    const { name: _name, description: _description, email: _email } = localInfo;

    setName(_name);
    setDescription(_description);
    setEmail(_email || '');
    setOriginalName(_name);
    setOriginalDescription(_description);
    setOriginalEmail(_email || '');
  };

  useEffect(() => {
    const localUploadInfoStr = localStorage.getItem('upload');

    if (localUploadInfoStr) {
      try {
        const localUploadInfo = JSON.parse(localUploadInfoStr);

        setInfo(localUploadInfo);
      } catch {
        console.error('Failed to parse local upload info');
      }
    }

    fetchLoadInfo();
  }, []);

  useEffect(() => {
    if (loadInfo) {
      setInfo(loadInfo);
    }
  }, [loadInfo]);

  const handleNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setName(e.target.value);
    checkIfEdited(e.target.value, description, email);
  };

  const handleDescriptionChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setDescription(e.target.value);
    checkIfEdited(name, e.target.value, email);
  };

  const validateEmail = (value: string) => {
    if (!value) {
      setEmailError('Email is required');

      return false;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    if (!emailRegex.test(value)) {
      setEmailError('Please enter a valid email address');

      return false;
    }

    setEmailError('');

    return true;
  };

  const handleEmailChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;

    setEmail(value);
    validateEmail(value);
    checkIfEdited(name, description, value);
  };

  const checkIfEdited = (newName: string, newDescription: string, newEmail: string) => {
    setIsEdited(
      newName !== originalName ||
        newDescription !== originalDescription ||
        newEmail !== originalEmail
    );
  };

  const handleSave = async () => {
    if (!name.trim()) {
      message.error('Name cannot be empty');

      return;
    }

    if (name.includes(' ')) {
      message.error('Name cannot contain spaces');

      return;
    }

    if (!validateEmail(email)) {
      return;
    }

    if (loadInfo) {
      // Update user information
      try {
        const res = await updateLoadInfo({ name, description, email });

        if (res.data.code === 0) {
          message.success('Identity updated successfully');
          // Update local storage

          const updatedData = { ...loadInfo, name, description, email };

          localStorage.setItem('upload', JSON.stringify(updatedData));

          useLoadInfoStore.getState().fetchLoadInfo();
        } else {
          message.error(res.data.message);
        }
      } catch (error) {
        console.error('Failed to update identity:', error);
        message.error('Failed to update identity');
      } finally {
        setIsEdited(false);
      }
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-6 py-8 space-y-8 overflow-y-auto h-full">
      {/* Page Title and Description */}
      <div className="mb-4">
        <h1 className="text-xl font-semibold text-gray-900 mb-1">{pageTitle}</h1>
        <p className="text-gray-600 max-w-3xl">{pageDescription}</p>
      </div>

      <div className="bg-white rounded-xl shadow-sm p-4">
        <div className="space-y-4">
          <div className="space-y-3">
            {/* Name section */}
            <div>
              <label className="block text-[14px] font-medium text-gray-700 mb-0.5">
                Second Me Name
              </label>
              <p className="text-sm text-gray-500 mb-1 leading-relaxed">
                This name will represent you, and your Second Me.
              </p>
              <input
                className="w-full px-4 py-2 border border-gray-300 rounded-lg bg-white text-gray-700 focus:border-gray-400 focus:ring-2 focus:ring-gray-400/20 transition-all shadow-[3px_3px_0px_0px_rgba(0,0,0,0.03)]"
                maxLength={20}
                onChange={handleNameChange}
                placeholder="e.g., Felix (no spaces allowed)"
                type="text"
                value={name}
              />
              <p className="mt-0.5 text-xs text-gray-500">{name.length}/20 characters</p>
            </div>

            <div>
              <label className="block text-[14px] font-medium text-gray-700 mb-0.5">
                Short Personal Description
              </label>
              <p className="text-sm text-gray-500 mb-1 leading-relaxed">
                Briefly describe yourself: personality, motivation, or style.
              </p>
              <textarea
                className="w-full px-4 py-2 border border-gray-300 rounded-lg bg-white text-gray-700 focus:border-gray-400 focus:ring-2 focus:ring-gray-400/20 transition-all shadow-[3px_3px_0px_0px_rgba(0,0,0,0.03)] resize-none leading-relaxed"
                maxLength={200}
                onChange={handleDescriptionChange}
                placeholder="e.g., 'An adventurous, data-driven, and enjoy learning new technologies.'"
                rows={3}
                value={description}
              />
              <p className="mt-0.5 text-xs text-gray-500">{description.length}/200 characters</p>
            </div>

            <div>
              <label className="block text-[14px] font-medium text-gray-700 mb-0.5">
                Email of Second Me
              </label>
              <p className="text-sm text-gray-500 mb-1 leading-relaxed">
                This email will be used as a contact point for your Second Me. You can use your own
                email address.
              </p>
              <input
                className={`w-full px-4 py-2 border ${emailError ? 'border-red-500' : 'border-gray-300'} rounded-lg bg-white text-gray-700 focus:border-gray-400 focus:ring-2 focus:ring-gray-400/20 transition-all shadow-[3px_3px_0px_0px_rgba(0,0,0,0.03)]`}
                onChange={handleEmailChange}
                placeholder="e.g., your.name@example.com"
                type="email"
                value={email}
              />
              {emailError && <p className="mt-1 text-xs text-red-500">{emailError}</p>}
            </div>

            <div className="flex justify-end pt-2">
              <button
                className={`px-6 py-2 rounded-lg text-white font-medium transition-all ${
                  isEdited
                    ? 'bg-blue-500 hover:bg-blue-600 cursor-pointer'
                    : 'bg-gray-300 cursor-not-allowed'
                }`}
                disabled={!isEdited}
                onClick={() => {
                  handleSave();
                }}
              >
                Save
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Next button outside the form */}
      <div className="mt-6 flex justify-end">
        <button
          className="px-4 py-2 text-sm font-medium rounded-lg bg-blue-500 text-white hover:bg-blue-600 transition-colors flex items-center gap-2"
          onClick={() => router.push(ROUTER_PATH.TRAIN_MEMORIES)}
        >
          Next: Upload Memories
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path d="M9 5l7 7-7 7" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} />
          </svg>
        </button>
      </div>
    </div>
  );
}
